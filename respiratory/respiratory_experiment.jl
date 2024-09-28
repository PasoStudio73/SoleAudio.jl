using Pkg
# Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.activate("/home/paso/Documents/Aclai/audio-rules2024")

using CSV, DataFrames, JLD2
using Audio911
using Catch22
using Random, StatsBase
using Plots

# include("../afe.jl")
include("../utils.jl")

# -------------------------------------------------------------------------- #
#                                 parameters                                 #
# -------------------------------------------------------------------------- #
propositional = false
# propositional = true

# audio extra parameters
sr = 8000
audioparams = (
    sr = sr,
    nfft = 256,
    mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = 26,
    mfcc_ncoeffs = 13,
    mel_freqrange = (300, round(Int, sr / 2)),
    featset = (:lin, :mfcc, :f0, :deltas)
)

keep_only = ["Healthy", "Pneumonia"]
# keep_only = ["Healthy", "COPD"]
# keep_only = ["Healthy", "URTI"]
# keep_only = ["Healthy", "Bronchiectasis"]
# keep_only = ["Healthy", "Bronchiolitis"]
# keep_only = ["Healthy", "URTI", "COPD", "Pneumonia"]

label = :diagnosis

# initialize random seed
seed = 11
Random.seed!(seed)
rng = MersenneTwister(seed)

dest_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/respiratory"

# -------------------------------------------------------------------------- #
#                             dataset setup utils                            #
# -------------------------------------------------------------------------- #
# rename labels based on patient id and diagnosis
function rename_labels!(df::DataFrame, csv_path::String)
    @info "Renaming labels..."
    labels = csv2dict(csv_path)

    insertcols!(df, 2, :diagnosis => fill(missing, nrow(df)), :id => fill(missing, nrow(df)))
    df[!, :diagnosis] = map(x -> labels[split(x, "_")[1]], df[!, :filename])
    df[!, :id] = map(x -> split(x, "_")[1], df[!, :filename])
end

function _audio_features(x::AbstractVector{Float64}, audioparams::NamedTuple, propositional::Bool)
    x_features = Audio911.afe(x; featset=(:mfcc), audioparams...)

    if propositional
        x_features = hcat(collect(map(func -> func(i), custom_catch) for i in eachcol(x_features))...)
    end

    nan_replacer!(x_features)
    return x_features
end

function _audio_features(x::AbstractVector{Float64}, sample_length::Int64, args...)
    x_start = sample(rng, 1:(size(x, 1)-sample_length+1))

    _audio_features(x[x_start:x_start+sample_length-1], args...)
end

# function afe(x::AbstractVector{<:AbstractFloat}, args...)
#     afe(Float64.(x), args...)
# end

function audio_features(df::DataFrame, audioparams::NamedTuple, sample_length::Int64, propositional::Bool)
    @info "Extracting audio features..."
    
    insertcols!(df, ncol(df)+1, :afe => fill(missing, nrow(df)))
    df[!, :afe] = map(x -> _audio_features(x, sample_length, audioparams, propositional), df[!, :audio])
end

nan_replacer!(x::AbstractArray{Float64}) = replace!(x, NaN => 0.0)

# -------------------------------------------------------------------------- #
#                       calculate best sample length                         #
# -------------------------------------------------------------------------- #
function calc_best_length(df::DataFrame)
    @info "Calculating best length..."
    if hasproperty(df, :length)
        df_lengths = df[!, :length]
    elseif hasproperty(df, :audio)
        df_lengths = size.(df[:, :audio], 1)
    else
        error("no method to determine audio length.")
    end

    # plot histogram
    histogram(df_lengths, bins=100, title="Sample length distribution", xlabel="length", ylabel="count")

    println("min length: ", minimum(df_lengths), "\nmax length: ", maximum(df_lengths), "\n")
    println("mean length: ", floor(Int, mean(df_lengths)), "\nmedian length: ", floor(Int, median(df_lengths)), "\n")

    h = fit(Histogram, df_lengths, nbins=100)
    max_index = argmax(h.weights)  # fet the index of the bin with the highest value
    # get the value of the previous hist bin of the one with highest value, to get more valid samples
    sample_length = round(Int64, h.edges[1][max_index == 1 ? max_index : max_index - 1])

    nsamples = size(df, 1)
    nvalid = size(filter(row -> row[:length] >= sample_length, df), 1)

    while (nvalid/nsamples) * 100 < 90. 
        sample_length -= 1
        nvalid = size(filter(row -> row[:length] >= sample_length, df), 1)
    end

    println("number of samples too short: ", nsamples - nvalid)
    println("remaining valid samples: ", nvalid)

    return sample_length
end

# -------------------------------------------------------------------------- #
#                                  folders                                   #
# -------------------------------------------------------------------------- #
csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/patient_diagnosis.csv"
wav_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/Respiratory_Sound_Database/audio_partitioned"

# -------------------------------------------------------------------------- #
#                                    main                                    #
# -------------------------------------------------------------------------- #
df = collect_audio_from_folder(wav_path, audioparams.sr)
rename_labels!(df, csv_path)

show_subdf(df, label)

df = balance_subdf(df, label, keep_only)

sample_length = calc_best_length(df)

df = filter(row -> row[:length] >= sample_length, df)
df = balance_subdf(df, label)

samples_proc = audio_features(df, audioparams, sample_length, propositional)

Y = df[:, [label, :id]]

X = DataFrame()

if !propositional
    X[!, label] = String[]
    X[!, :id] = String[]
    for i in 1:size(samples_proc[1], 2)
        colname = "a$i"
        X[!, colname] = Vector{Float64}[]
    end
end

for i in axes(Y, 1)
    push!(X, vcat(Y[i, :]..., vcat.(k for k in eachcol(samples_proc[i]))))
end

fname = string("/respiratory_", keep_only[2], "_$(audioparams.mel_scale)_$(audioparams.mel_nbands)")
save_jld2(X, string(dest_path, fname, ".jld2"); labels=size(Y, 2))