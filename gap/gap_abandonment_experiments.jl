using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DataFrames, CSV, JLD2
using Audio911
using Plots
using StatsBase
using Random

include("../afe.jl")
include("../utils.jl")

# initialize random seed
seed = 11
Random.seed!(seed)
rng = MersenneTwister(seed)

#--------------------------------------------------------------------------------------#
#                                         utils                                        #
#--------------------------------------------------------------------------------------#
filename = (x) -> (string("Example_", x, ".wav"))

function age2string(x_birth::AbstractString)
    x_birth = parse(Int, x_birth[1:4])
    2023 - x_birth[1] <= 50 && return "young"
    return "old"
end

function balancing_groups(df::DataFrame, label::Symbol)
    # balancing groups
    grouped_df = groupby(df, label)
    if length(grouped_df) > 2
        error("There's more than 2 groups, something went wrong.")
    end
    if size(grouped_df[1], 1) < size(grouped_df[2], 1)
        max_samples = size(grouped_df[1], 1)
        group_1 = grouped_df[1]
        group_2 = grouped_df[2][1:max_samples, :]
    else
        max_samples = size(grouped_df[2], 1)
        group_1 = grouped_df[1][1:max_samples, :]
        group_2 = grouped_df[2]
    end
    println("Max samples available per group: $max_samples")

    combined_df = vcat(group_1, group_2)
end

function push_wavs(df_info::DataFrame)
    for i in eachrow(df_info)
        wavfile = joinpath(source_wav_path, filename(i.filename))
        x = load_audio(file=wavfile, sr=sr, norm=true)
        x = speech_detector(audio=x, sr=x.sr, overlap_length=overlap_length=round(Int, 0.02 * x.sr))

        n_path = i.outcome == "data_collection_completed" ? wav_path[2] : wav_path[3]

        save_wavs && save_audio(joinpath(n_path, filename(i.filename)), x.data, x.sr)
    end
end

# -------------------------------------------------------------------------- #
#                            build working dataframe                         #
# -------------------------------------------------------------------------- #
function build_df(source_db::String)
    @info "Starting..."

    # load dataset
    df_info = CSV.read(source_db, DataFrame)
    rename!(df_info, :payee_data_name => :name)
    # rename!(df_info, :payee_data_birthdate => :birth)
    rename!(df_info, :ultimo_esito_cciad => :outcome)

    size_1 = size(df_info, 1)
    println("Entries present in dataset: $size_1")

    # filter out missing outcome labels
    df_info = filter(row -> !ismissing(row.outcome) && (row.outcome != "data_collection_completed" || row.outcome != "info_endpoint"), df_info)

    # filter missing wav files
    df_info = filter(i -> isfile(joinpath(source_wav_path, filename(i.filename))), df_info)

    combined_df = balancing_groups(df_info, :outcome)

    select(combined_df, [:filename, :outcome])
end

# -------------------------------------------------------------------------- #
#                          sample length functions                           #
# -------------------------------------------------------------------------- #
function analyse_lengths(X_lengths::AbstractVector{Int64})
    @info "Analyse sample length..."
    # plot histogram
    histogram(X_lengths, bins=100, title="Sample length distribution", xlabel="length", ylabel="count")

    println("min length: ", minimum(X_lengths), "\nmax length: ", maximum(X_lengths))
    println("mean length: ", mean(X_lengths), "\nmedian length: ", median(X_lengths))

    h = fit(Histogram, X_lengths, nbins=100)
    max_index = argmax(h.weights)  # fet the index of the bin with the highest value
    # get the value of the previous hist bin of the one with highest value, to get more valid samples
    sample_length = round(Int64, h.edges[1][max_index])

    println("\nsuggested length: ", sample_length)
    not_valid = length(filter(x -> x < sample_length, X_lengths))
    println("number of samples too short: ", not_valid)
    println("remaining valid samples: ", length(X_lengths) - not_valid)

    return sample_length
end

# -------------------------------------------------------------------------- #
#                           extract audio features                           #
# -------------------------------------------------------------------------- #
function extract_audio_features!(
    df_info::DataFrame,
    sr::Int64,
    audio_set::NamedTuple,
    wavsample_path::Dict,
    features_set::Symbol
)
    @info "Extract audio features..."

    X = Vector{Float64}[]
    X_lengths = Int64[]

    for i in eachrow(df_info)
        wavfile = string(wavsample_path[i.outcome], "/", filename(i.filename))
        x, _ = load_audio(wavfile, sr)
        # normalize and speech detector already done
        push!(X, Float64.(x))
        push!(X_lengths, length(x))
    end

    sample_length = analyse_lengths(X_lengths)

    X_feats = DataFrame()

    for i in 1:nrow(df_info)
        valid_length = length(X[i]) - sample_length
        if valid_length >= 0
            random_start = sample(rng, 1:valid_length)
            if features_set != :wavelets
                x_obj = audio_obj(X[i][random_start:random_start+sample_length], sr; audio_set...)
                x_features = get_features(x_obj, features_set)
            else
                x_features = get_wpspec(X[i][random_start:random_start+sample_length], sr)
            end

            nan_replacer!(x_features)
            nan_buster(x_features)

            if isempty(X_feats)
                # initialize dataframe
                X_feats[!, :outcome] = String[]
                for i in 1:size(x_features, 2)
                    colname = "a$i"
                    X_feats[!, colname] = Vector{Float64}[]
                end
            end
            push!(X_feats, vcat(df_info[i, :outcome], vcat.(k for k in eachcol(x_features))))
        end
    end

    X_feats = balancing_groups(X_feats, :outcome)

    return X_feats
end

# -------------------------------------------------------------------------- #
#                                 parameters                                 #
# -------------------------------------------------------------------------- #
source_db = "/home/paso/datasets/GAP/anonymized_dataset_fixed/datasetinfo.csv"

source_wav_path = "/home/paso/datasets/GAP/anonymized_dataset_fixed"
# write_missing = true
write_missing = false

labels_db = ["path", "name", "gender", "outcome"]

name_db = "/home/paso/results/speech/data/map_name_abandonment.csv"

# save_csv = true
save_csv = false

df_csv_path = "/home/paso/datasets/GAP/results"
labels_csv = ["filename", "outcome"]

# save_wavs = true
save_wavs = false
wav_path = (
    "/home/paso/datasets/GAP/wavfiles",
    "/home/paso/datasets/GAP/wavfiles/outcome/positive",
    "/home/paso/datasets/GAP/wavfiles/outcome/negative",
)

plot_durations = false

wavsample_path = Dict(
    "data_collection_completed" => "/home/paso/datasets/GAP/wavfiles/outcome/positive",
    "info_endpoint" => "/home/paso/datasets/GAP/wavfiles/outcome/negative",
)

features_set = :full
# features_set = :gap
# features_set = :wavelets

feature_df_path = "/home/paso/datasets/GAP/jld2"
# feature_df_name = "abandonment_full"
feature_df_name = string("gap_abandonment_", features_set)

# audio settings
sr = 8000 # database sample rate
stft_length = 256

audio_set = (
    # fft
    stft_length=stft_length,

    # spectrum
    freq_range=(0, floor(Int, sr / 2)),
)

# -------------------------------------------------------------------------- #
#                                   main                                     #
# -------------------------------------------------------------------------- #
if save_csv
    save_wavs && (isdir.(wav_path) .|| mkpath.(wav_path))

    df_info = build_df(source_db)
    CSV.write(
        string(df_csv_path, "/GAP_dataset_abandonment.csv"),
        df_info,
        writeheader=true,
        header=labels_csv
    )

    save_wavs && push_wavs(df_info)
end

df_info = DataFrame(CSV.File(string(df_csv_path, "/GAP_dataset_abandonment.csv")))
X_feats = extract_audio_features!(df_info, sr, audio_set, wavsample_path, features_set)
save_jld2(X_feats, string(feature_df_path, "/", feature_df_name, ".jld2"))
