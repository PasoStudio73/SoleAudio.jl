using Pkg
# Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.activate("/home/paso/Documents/Aclai/audio-rules2024")

using Revise
using CSV, DataFrames, JLD2
using Audio911
using Catch22
using StatsBase
using Plots

include("../utils.jl")

# -------------------------------------------------------------------------- #
#                                parameters                                  #
# -------------------------------------------------------------------------- #
sr = 8000
audioparams = (
    sr = sr,
    nfft = 256,
    mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = 26,
    mfcc_ncoeffs = 13,
    mel_freqrange = (0, round(Int, sr / 2)),
    featset = (:mfcc, :deltas)
)

min_length = 8000
min_samples = 400

wav_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"

# source_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"
# dest_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/emotion"
# wav_jld2_name = "ravdess_wavfiles"

classes = :emo2bins
# classes = :emo4bins
# classes = :emo8bins

if classes == :emo2bins
    classes_dict = Dict{String,String}(
        "01" => "positive",
        "02" => "positive",
        "03" => "positive",
        "04" => "negative",
        "05" => "negative",
        "06" => "negative",
        "07" => "negative",
        "08" => "positive"
    )
# elseif classes == :emo4bins
elseif classes == :emo8bins
    classes_dict = Dict{String,String}(
        "01" => "neutral",
        "02" => "calm",
        "03" => "happy",
        "04" => "sad",
        "05" => "angry",
        "06" => "fearful",
        "07" => "disgust",
        "08" => "surprised"
    )
else
    error("Unknown class $classes")
end

classes_func(row) = match(r"^(?:[^-]*-){2}([^-]*)", row.filename)[1]


# label = :emotion

# -------------------------------------------------------------------------- #
#                          sample length functions                           #
# -------------------------------------------------------------------------- #
function analyse_lengths(X_lengths::AbstractVector{Int64})
    # plot histogram
    histogram(X_lengths, bins=100, title="Sample length distribution", xlabel="length", ylabel="count")

    println("min length: ", minimum(X_lengths), "\nmax length: ", maximum(X_lengths))
    println("mean length: ", mean(X_lengths), "\nmedian length: ", median(X_lengths))

    h = fit(Histogram, X_lengths, nbins=100)
    max_index = argmax(h.weights)  # fet the index of the bin with the highest value
    # get the value of the previous hist bin of the one with highest value, to get more valid samples
    return round(Int64, h.edges[1][max_index == 1 ? max_index : max_index - 1])
end

function collect_files(ds_params::DsParams)
    @info "Collect files..."
    # initialize id path dataframe
    file_set = DataFrame(id=String[], length=Int64[], x=AbstractArray{<:AbstractFloat}[])

    # collect files
    for single_sourcepath in ds_params.source_path
        for (root, _, files) in walkdir(single_sourcepath)
            for i in files[occursin.(".wav", files).|occursin.(".flac", files).|occursin.(".mp3", files)]
                audio = load_audio(wavfile=joinpath(root, i), sr=ds_params.sr)
                audio = speech_detector(audio=audio.data, sr=audio.sr, overlap_length=round(Int, 0.02 * sr))
                push!(file_set, hcat(i, size(audio.data, 1), [audio.data]))
            end
        end
    end

    @info "Starting check dataset files length..."

    if ds_params.fixed_length == 0.0
        sample_length = analyse_lengths(file_set[!, :length])
    else
        sample_length = round(Int, ds_params.fixed_length * sr)
    end

    println("\nsuggested length: ", sample_length, " around: ", sample_length / sr, " sec.")
    valid_set = filter(row -> row[:length] >= sample_length, file_set)

    nvalid = size(valid_set, 1)
    println("number of samples too short: ", size(file_set, 1) - nvalid)
    println("remaining valid samples: ", nvalid)

    return valid_set, sample_length
end

# -------------------------------------------------------------------------- #
#                     collect files and build dataframe                      #
# -------------------------------------------------------------------------- #
function build_set(wavdf::DataFrame, sample_length::Int64, ds_params::DsParams)
    # create dataframe for audio features storing
    X = DataFrame()

    for i in eachrow(wavdf)
        # get id labels
        fileinfo = filter(x -> !isempty(x) && all(isdigit, x), split(i[:id], r"(\D+)"))[3]

        label = ds_params.emotion_set[fileinfo[3]]
        # println(label)

        valid_length = length(i[:x]) - sample_length
        if valid_length >= 0
            random_start = sample(rng, 1:valid_length)
            # println(typeof(i[:x][random_start:random_start+sample_length]))
            x_features = audio911_extractor(i[:x][random_start:random_start+sample_length])
            nan_replacer!(x_features)
        end

        if isempty(X)
            # initialize dataframe
            X[!, ds_params.label] = String[]
            if ds_params.propositional
                for i in 1:length(x_features)
                    colname = "a$i"
                    X[!, colname] = Float64[]
                end
            else
                for i in 1:size(x_features, 2)
                    colname = "a$i"
                    X[!, colname] = Vector{Float64}[]
                end
            end
        end

        if ds_params.propositional
            push!(X, vcat(label, x_features...))
        else
            push!(X, vcat(label, vcat.(k for k in eachrow(x_features'))))
        end
    end

    return X
end

# -------------------------------------------------------------------------- #
#                              balance classes                               #
# -------------------------------------------------------------------------- #
function balance(df::DataFrame, ds_params::DsParams)
    @info "Balancing classes..."

    grp_df = groupby(df, ds_params.label)
    # Find the group with the shortest length
    n_samples = minimum(size(i, 1) for i in grp_df)

    X = DataFrame()
    for i in grp_df
        choosen_indexes = sample(rng, 1:nrow(i), n_samples, replace=false)
        for index in choosen_indexes
            push!(X, i[index, :])
        end
    end

    return X
end

# -------------------------------------------------------------------------- #
#                                   main                                     #
# -------------------------------------------------------------------------- #
jld2_file = string(wav_jld2_path, "/", wav_jld2_name, ".jld2")
# if store_wav_jld2
#     wavdf, sample_length = collect_files(ds_params)
#     save_wav_jld2(wavdf, sample_length, jld2_file)
# else
    wavdf, sample_length = load_wav_jld2(jld2_file)
# end

df = build_set(wavdf, sample_length, ds_params)

df = balance(df, ds_params)

ds_name = string("emods_", classes, "_", propositional ? "prop" : "modal")
save_jld2(df, string(dest_path, "/", ds_name, ".jld2"))

@info "Done."

function _collect_audio_from_folder!(df::DataFrame, path::String, sr::Int64; norm::Bool=true, spc_detect::Bool=false)
    # collect files
    for (root, _, files) in walkdir(path)
        for file in filter(f -> any(occursin.([".wav", ".flac", ".mp3"], f)), files)
            audio = load_audio(; source=joinpath(root, file), sr=sr, norm=norm)
            spc_detect && (audio = speech_detector(source=audio))
            push!(df, hcat(split(file, ".")[1], size(audio.data, 1), [audio.data]))
        end
    end
end

function collect_audio_from_folder(path::String; sr::Int64, kwargs...)
    @info "Collect files..."
    # initialize id path dataframe
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
    _collect_audio_from_folder!(df, path, sr; kwargs...)
    return df
end

function collect_audio_from_folder(path::AbstractVector{String}; sr::Int64, kwargs...)
    @info "Collect files..."
    df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])

    for i in path
        _collect_audio_from_folder!(df, i, sr; kwargs...)
    end

    return df
end

df = collect_audio_from_folder(wav_path; sr=audioparams.sr, norm=true, spc_detect=true)
csvdf = read_filenames(df, classes_func, classes_dict)
merge_df_csv!(df, csvdf)

trm_df = trimlength_df(df, :label, :length, :audio; min_length=min_length, min_samples=min_samples)
afe_df = audio_features(trm_df, audioparams)


