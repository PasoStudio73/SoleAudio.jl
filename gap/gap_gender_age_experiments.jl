using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DataFrames, CSV, JLD2
using Audio911
using Plots
using StatsBase

include("../afe.jl")
include("../utils.jl")

#--------------------------------------------------------------------------------------#
#                                         utils                                        #
#--------------------------------------------------------------------------------------#
function nan_buster(x_features::Matrix{Float64})
    any(isnan, x_features) && @warn("Found NaN value!")
end

function nan_replacer!(x_features::Matrix{<:AbstractFloat})
    count = Base.count(isnan, x_features)
    replace!(x_features, NaN => 0.0)
    # println("found $count nan values.")
    return x_features
end

function age2string(x_birth::AbstractString)
    x_birth = parse(Int, x_birth[1:4])
    2023 - x_birth[1] <= 50 && return "young"
    return "old"
end

function load_jld2(filename::String; append=false)
    append == true ? t = "a+" : t = "r"

    d = jldopen(filename, t)
    df, Y = d["dataframe_validated"]

    @assert df isa DataFrame

    return df, Y
end

function save_jld2(df_info::DataFrame, X::DataFrame, jld2_file::String)
    @info "Save jld2 file..."

    # df = X[:, 2:end]
    # y = X[:, 1]

    dataframe_validated = (X, df_info)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    # println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function push_wavs(df_info::DataFrame)
    for i in eachrow(df_info)
        wavfile = joinpath(source_wav_path, filename(i.filename))
        x, _ = load_audio(wavfile, sr)
        x = normalize_audio(x)
        x, _ = speech_detector(x, sr, overlap_length=round(Int, 0.02 * sr))

        n_path = i.gender == "f" ? (i.age == "young" ? wav_path[2] : wav_path[3]) : (i.age == "young" ? wav_path[4] : wav_path[5])

        save_wavs && save_audio(joinpath(n_path, filename(i.filename)), x, sr)
    end
end

filename = (x) -> (string("Example_", x, ".wav"))

# -------------------------------------------------------------------------- #
#                                test and debug                              #
# -------------------------------------------------------------------------- #
function find_missing(df_info::DataFrame, source_wav_path::String, write_missing::Bool)
    other_folders = [
        "/home/paso/datasets/GAP/anonymized_dataset",
        "/home/paso/datasets/GAP/anonymized_dataset_fixed",
        "/home/paso/datasets/GAP/anonymized_dataset_fixed_partitioned"
    ]

    println("\nSeraching missing files...")
    if write_missing
        txtresults = open(missing_files_txt, "w")
        println(txtresults, "GAP dataset missing files:\n")
    end

    found_something = false
    for j in other_folders
        for i in eachrow(df_info)
            file_name = filename(i.filename)
            if !isfile(joinpath(source_wav_path, file_name))
                for j in other_folders
                    if isfile(joinpath(j, file_name))
                        write_missing && println(file_name, "found in: $j")
                        found_something = true
                    else
                        if write_missing
                            println(txtresults, file_name)
                        end
                    end
                end
            end
        end
    end

    if !found_something
        println("No missing files match found.")
    end
    if write_missing
        close(txtresults)
    end
end

# -------------------------------------------------------------------------- #
#                            build working dataframe                         #
# -------------------------------------------------------------------------- #
function build_df(source_db::String, names_db::String, write_missing::Bool)
    @info "Starting..."

    # load dataset
    df_info = CSV.read(source_db, DataFrame)
    rename!(df_info, :payee_data_name => :name)
    rename!(df_info, :payee_data_birthdate => :birth)
    size_1 = size(df_info, 1)
    println("Entries present in dataset: $size_1")

    # filter out missing and not recognized name labels
    df_info = filter(row -> !ismissing(row.name) && row.name != "not recognized name", df_info)
    size_2 = size(df_info, 1)
    println("Entries whitout a valid name: $(size_1 - size_2), remaning valid: $size_2")
    df_info = filter(row -> !ismissing(row.birth), df_info)
    size_3 = size(df_info, 1)
    println("Entries whitout a valid birthdate: $(size_2 - size_3), remaning valid: $size_3")
    # select only useful colums
    df_info = select(df_info, [:filename, :name, :birth])

    find_missing(df_info, source_wav_path, write_missing)

    df_info = filter(i -> isfile(joinpath(source_wav_path, filename(i.filename))), df_info)
    size_4 = size(df_info, 1)
    println("\ntotal missing files: $(size_3 - size_4), remaning: $size_4 files\n")

    # load dataset name and gender
    name2gender = CSV.File(names_db) |> Dict
    if write_missing
        txtresults = open(missing_names_txt, "w")
        println(txtresults, "GAP dataset missing names:\n")
        for i in eachrow(df_info)
            if !haskey(name2gender, i.name)
                println(txtresults, i.name)
            end
        end
        close(txtresults)
    end

    insertcols!(df_info, :gender => [get(name2gender, df_info[i, :name], "unknown") for i in 1:nrow(df_info)])
    # filter unknown gender
    df_info = filter(row -> row.gender != "unknown", df_info)
    size_5 = size(df_info, 1)
    println("Entries whitout known name: $(size_4 - size_5), remaning $size_5 valid files.")

    # calc age
    insertcols!(df_info, :age => [age2string(i.birth) for i in eachrow(df_info)])

    select(df_info, [:filename, :gender, :age])
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
    sample_length = round(Int64, h.edges[1][max_index == 1 ? max_index : max_index - 1])

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
)
    @info "Extract audio features..."

    X = Vector{Float64}[]
    X_lengths = Int64[]

    for i in eachrow(df_info)
        wavfile = string(wavsample_path[i.gender], "/", wavsample_path[i.age], "/", filename(i.filename))
        x, _ = load_audio(wavfile, sr)
        # normalize and speech detector already done
        push!(X, Float64.(x))
        push!(X_lengths, length(x))
    end

    sample_length = analyse_lengths(X_lengths)

    X_feats = DataFrame()
    deleted_rows = 0

    for i in 1:nrow(df_info)
        if length(X[i]) >= sample_length
            x_obj = audio_obj(X[i][1:sample_length], sr; audio_set...)
            x_features = get_features(x_obj, :gap)

            nan_replacer!(x_features)
            nan_buster(x_features)

            if isempty(X_feats)
                # initialize dataframe
                for i in 1:size(x_features, 2)
                    colname = "a$i"
                    X_feats[!, colname] = Vector{Float64}[]
                end
            end
            push!(X_feats, vcat.(k for k in eachcol(x_features)))
        else
            delete!(df_info, [i - deleted_rows])
            deleted_rows += 1
        end
    end

    return X_feats
end

# -------------------------------------------------------------------------- #
#                                 parameters                                 #
# -------------------------------------------------------------------------- #
source_db = "/home/paso/datasets/GAP/anonymized_dataset_fixed/datasetinfo.csv"
source_wav_path = "/home/paso/datasets/GAP/anonymized_dataset_fixed"

labels_db = ["path", "name", "gender", "age"]

names_db = "/home/paso/results/speech/data/map_name_gender.csv"

# write_missing = true
write_missing = false
missing_files_txt = "/home/paso/datasets/GAP/results/missing_files.txt"
missing_names_txt = "/home/paso/datasets/GAP/results/missing_names.txt"

# save_csv = true
save_csv = false
df_csv_path = "/home/paso/datasets/GAP/results"
labels_csv = ["filename", "gender", "age"]

# save_wavs = true
save_wavs = false
wav_path = (
    "/home/paso/datasets/GAP/wavfiles",
    "/home/paso/datasets/GAP/wavfiles/female/young",
    "/home/paso/datasets/GAP/wavfiles/female/old",
    "/home/paso/datasets/GAP/wavfiles/male/young",
    "/home/paso/datasets/GAP/wavfiles/male/old",
)

plot_durations = false

wavsample_path = Dict(
    "f" => "/home/paso/datasets/GAP/wavfiles/female",
    "m" => "/home/paso/datasets/GAP/wavfiles/male",
    "young" => "young",
    "old" => "old"
)

feature_df_path = "/home/paso/datasets/GAP/jld2"
feature_df_name = "features_mfcc_f0"

# audio settings
sr = 8000 # database sample rate
fft_length = 256

audio_set = (
    # fft
    fft_length=fft_length,
    window_type=(:hann, :periodic),
    window_length=fft_length,     # standard setting: round(Int, 0.03 * sr)
    overlap_length=round(Int, fft_length / 2),    # standard setting: round(Int, 0.02 * sr)
    window_norm=false,

    # spectrum
    frequency_range=(0, floor(Int, sr / 2)),
    spectrum_type=:power,     # :power, :magnitude

    # mel
    mel_style=:htk,     # :htk, :slaney, :tuned
    mel_bands=26,
    filterbank_design_domain=:linear,
    filterbank_normalization=:bandwidth,     # :bandwidth, :area, :none
    frequency_scale=:mel,

    # mfcc
    num_coeffs=13,
    normalization_type=:dithered,     # :standard, :dithered
    rectification=:log,     # :log, :cubic_root
    log_energy_source=:standard,     # :standard (after windowing), :mfcc
    log_energy_pos=:none,     #:append, :replace, :none
    delta_window_length=9,
    delta_matrix=:transposed,     # :standard, :transposed

    # spectral
    spectral_spectrum=:lin,     # :lin, :mel

    # f0
    f0_method=:nfc,
    f0_range=(50, 400),
    median_filter_length=1,
)

# -------------------------------------------------------------------------- #
#                                   main                                     #
# -------------------------------------------------------------------------- #
if save_csv
    save_wavs && (isdir.(wav_path) .|| mkpath.(wav_path))

    df_info = build_df(source_db, names_db, write_missing)
    CSV.write(
        string(df_csv_path, "/GAP_dataset.csv"),
        df_info,
        writeheader=true,
        header=labels_csv
    )

    save_wavs && push_wavs(df_info)
end

df_info = DataFrame(CSV.File(string(df_csv_path, "/GAP_dataset.csv")))
X_feats = extract_audio_features!(df_info, sr, audio_set, wavsample_path)
save_jld2(df_info, X_feats, string(feature_df_path, "/", feature_df_name, ".jld2"))
