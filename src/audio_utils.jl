# using CSV, DataFrames, JLD2
# using Audio911
# using ModalDecisionTrees
# using SoleDecisionTreeInterface
# using MLJ, Random
# using StatsBase, Catch22
# using CategoricalArrays
# using Plots

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
catch9_f = ["max", "min", "mean", "med", "std", "bsm", "bsd", "qnt", "3ac"]

color_code = Dict(:red => 31, :green => 32, :yellow => 33, :blue => 34, :magenta => 35, :cyan => 36)
r_select = r"\e\[\d+m(.*?)\e\[0m"
r_split = r"(\e\[[\d;]*m)(.*?)(\e\[0m)"

function vnames_builder(featset::Tuple, audioparams::NamedTuple; freq::AbstractVector{Int})
    vcat(
        :mel in featset ? ["\e[$(color_code[:yellow])mmel$i=$(freq[i])Hz\e[0m" for i in 1:audioparams.mel_nbands] : String[],
        :mfcc in featset ? ["\e[$(color_code[:red])mmfcc$i\e[0m" for i in 1:audioparams.mfcc_ncoeffs] : String[],
        :f0 in featset ? ["\e[$(color_code[:green])mf0\e[0m"] : String[],
        :spectrals in featset ? begin
            vcat("\e[$(color_code[:cyan])mcntrd\e[0m", "\e[$(color_code[:cyan])mcrest\e[0m",
            "\e[$(color_code[:cyan])mentrp\e[0m", "\e[$(color_code[:cyan])mflatn\e[0m", "\e[$(color_code[:cyan])mflux\e[0m",
            "\e[$(color_code[:cyan])mkurts\e[0m", "\e[$(color_code[:cyan])mrllff\e[0m", "\e[$(color_code[:cyan])mskwns\e[0m",
            "\e[$(color_code[:cyan])mdecrs\e[0m", "\e[$(color_code[:cyan])mslope\e[0m", "\e[$(color_code[:cyan])msprd\e[0m")
        end : String[],
    )
end

# ---------------------------------------------------------------------------- #
#                             collect audio files                              #
# ---------------------------------------------------------------------------- #
params(x::Dict, audioparams::NamedTuple) = (x[k] => getfield(audioparams, Symbol(k)) for k in keys(x) if haskey(audioparams, Symbol(k)))

function walk_audio_dir!(df::DataFrame, path::String; audioparams::NamedTuple)
    norm = get(audioparams, :norm, false)
    speech_detect = get(audioparams, :speech_detect, false)
    sd_params = Dict(:sd_thr => :thresholds, :sd_spread_thr => :spread_threshold)

    for (root, _, files) in walkdir(path)
        for file in filter(f -> any(occursin.([".wav", ".flac", ".mp3"], f)), files)
            audio = load_audio(joinpath(root, file), audioparams.sr; norm=norm)
            speech_detect && begin audio, _ = speech_detector(audio; params(sd_params, audioparams)...) end
            push!(df, hcat(split(file, ".")[1], size(audio.data, 1), [audio.data]))
        end
    end
end

function _collect_audio_from_folder!(
    df::DataFrame, 
    path::String; 
    audioparams::NamedTuple,
    fragmented::Bool=false,
    frag_func::Union{Function, Nothing}=nothing,
)
    if !fragmented
        walk_audio_dir!(df, path; audioparams=audioparams)
    else
        !isnothing(frag_func) || throw(ArgumentError("`frag_func` must be provided when `fragmented=true`"))

        @info "Defragmenting audio files..."
        dff = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
        walk_audio_dir!(dff, path; audioparams=audioparams)

        dff.group = map(x -> split(x, "_")[1], dff.filename)
        grouped_dff = groupby(dff, :group)

        for i in grouped_dff
            audiodata = vcat(i.audio...)
            push!(df, hcat(frag_func(i.filename[1]), size(audiodata, 1), [audiodata]))
        end
    end
end

function collect_audio_from_folder(path::String; kwargs...)
    @info "Collect files..."
    # initialize id path dataframe
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
    _collect_audio_from_folder!(df, path; kwargs...)
    return df
end

function collect_audio_from_folder(path::AbstractVector{String}; kwargs...)
    @info "Collect files..."
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])

    for i in path
        _collect_audio_from_folder!(df, i; kwargs...)
    end
    
    return df
end

# ---------------------------------------------------------------------------- #
#                                 audio utils                                  #
# ---------------------------------------------------------------------------- #
function set_audio_length!(df::DataFrame, audiolabel::Symbol, lengthlabel::Symbol)
    insertcols!(df, lengthlabel => size.(df[:, audiolabel], 1))
end

function afe(
    df::DataFrame, 
    featset::Tuple, 
    audioparams::NamedTuple;
    label::Symbol=:label,
    source_label::Symbol=:audio, 
)
    @info("Collect audio features...")

    freq = round.(Int, audio_features(df[1, source_label], audioparams.sr; featset=(:get_only_freqs), audioparams...))
    variable_names = vnames_builder(featset, audioparams; freq=freq)

    # audiofeats = [audio_features(row[source_label], audioparams.sr; featset=featset, params=audioparams) for row in eachrow(df)]
    # X = DataFrame(label => String[])
    # for name in [match(r_select, v)[1] for v in variable_names]
    #     X[!, name] = AbstractVector{<:AbstractFloat}[]
    # end
    # for (i, x) in enumerate(audiofeats)
    #     push!(X, (df[i, label], eachcol(x)...))
    # end

    X = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])
    for row in eachrow(df)
        audiofeats = collect(eachcol(audio_features(row[source_label], audioparams.sr; featset=featset, audioparams...)))
        push!(X, audiofeats)
    end

    return X, CategoricalArray(df[!, label]), variable_names
end
