using Audio911
using Glob
using DataFrames, JLD2
using StatsBase
# using CSV

# using ConfigEnv
# dotenv()

#--------------------------------------------------------------------------------------#
#                                     parameters                                       #
#--------------------------------------------------------------------------------------#
sourcepath = [
    # "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/test",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f1",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f2",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f3",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m1",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m2",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m3",
]
# sourcepath = ENV["source_db"]

jld2_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/jld2/"
# dest_db = ENV["dest_db"]

# profile = :full
profile = :emotion

sr = 8000 # database sample rate
stft_length = 256

mood = Dict{String,String}(
    "neu" => "neu",
    "dis" => "neg",
    "pau" => "neg",
    "rab" => "neg",
    "tri" => "neg",
    "gio" => "pos",
    "sor" => "pos"
)

gender = Dict{String,String}("f" => "female", "m" => "male")

#--------------------------------------------------------------------------------------#
#                                  utility and regex                                   #
#--------------------------------------------------------------------------------------#
num_hops = (x, window_length, hop_length) -> floor(Int, (x - window_length) / hop_length) + 1
hop_length = (x, window_length, num_hops) -> floor(Int, (x - window_length) / (num_hops - 1))

fileregex = r"(^[a-zA-Z]*)-([a-zA-Z]*)"

function save_jld2(X::DataFrame, jld2_file::String)
    @info "Save jld2 file..."

    df = X[:, 4:end]
    y = X[:, 1:3]

    dataframe_validated = (df, y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function nan_buster(x_features::Matrix{<:AbstractFloat}, file::AbstractString)
    for i in axes(x_features, 1)
        if any(isnan, x_features[i, :])
            @warn("Found NaN value in feature: A$i, in file: $file.")
        end
    end
end

function nan_replacer!(x_features::Matrix{<:AbstractFloat})
    count = Base.count(isnan, x_features)
    replace!(x_features, NaN => 0.0)
    println("found $count nan values.")
    return x_features
end

#--------------------------------------------------------------------------------------#
#                            define fixed windows number                               #
#--------------------------------------------------------------------------------------#
function check_dataset_files(sourcepath::Vector{String}, sr::Int64, stft_length::Int64)

    function collect_files_length(
        source_path::Vector{String},
        sr::Int64
    )
        # initialize id path dataframe
        file_set = DataFrame(id=String[], length=Int64[])

        # collect ids
        for single_source_path in source_path
            for (root, _, files) in walkdir(single_source_path)
                for i in files[occursin.(".wav", files).|occursin.(".flac", files).|occursin.(".mp3", files)]
                    # println(joinpath(root, i))
                    file_name = joinpath(root, i)
                    x, sr = load_audio(file_name, sr=sr)
                    # da verificare
                    # speech_detector!(x, sr)
                    # lunghezza audio specificata in samples
                    push!(file_set, vcat(file_name, size(x, 1)))
                end
            end
        end

        return file_set
    end

    function calc_n_windows(
        x_min::Int64,
        x_max::Int64,
        stft_length::Int64
    )
        best_max = num_hops(x_max, stft_length, floor(Int, stft_length / 2))
        println("finestre per coprire al meglio il sample più lungo: ", best_max)

        hop_min = hop_length(x_min, stft_length, best_max)
        println("avanzamento nel sample più corto: ", hop_min)
        println("equivalente al: ", hop_min * 100 / stft_length, "%\n")

        # matlab_window_length::Int64 = round(Int, 0.03 * sr)
        matlab_hop_length::Int64 = round(Int, stft_length - 0.025 * sr)

        best_min = num_hops(x_min, stft_length, matlab_hop_length)
        println("finestre per coprire al meglio il sample più corto: ", best_min)

        hop_max = hop_length(x_max, stft_length, best_min)
        println("avanzamento nel sample più lungo: ", hop_max)
        println("equivalente al: ", hop_max * 100 / stft_length, "%\n")

        n_windows = ceil(Int, (best_max + best_min) / 2)
        println("numero finestre proposto: ", n_windows)

        return n_windows
    end
    @info "Starting check dataset files length..."

    file_set = collect_files_length(sourcepath, sr)
    println("total amount of audio samples: ", size(file_set, 1))

    x_mean = floor(Int, mean(file_set[!, :length]))
    x_median = floor(Int, median(file_set[!, :length]))
    x_min = minimum(file_set[!, :length])
    x_max = maximum(file_set[!, :length])

    println("audio lengths: mean: ", x_mean, ",\tmedian: ", x_median)
    println("               minimum: ", x_min, ",\tmaximum: ", x_max, "\n")

    n_windows = calc_n_windows(x_min, x_max, stft_length)
end
#--------------------------------------------------------------------------------------#
#                               extract audio features                                 #
#--------------------------------------------------------------------------------------#
function extract_audio_features(x::Vector{<:AbstractFloat}, sr::Int64, stft_length::Int64, overlap_length::Int64)
    audio_obj = audio_features_obj(x, sr; stft_length=stft_length, overlap_length=overlap_length)
    return audio_obj.get_features(profile)
end

#--------------------------------------------------------------------------------------#
#                          collect files and build dataframe                           #
#--------------------------------------------------------------------------------------#
function build_dataframe(sourcepath::Vector{String}, sr::Int64, stft_length::Int64, n_windows::Int64)
    # create dataframe for audio features storing
    ds_labels = [:mood, :emotion, :gender]
    X = DataFrame()

    # collect audio files
    for single_sourcepath in sourcepath
        for (root, _, files) in walkdir(single_sourcepath)
            for i in files[occursin.(".wav", files).|occursin.(".flac", files).|occursin.(".mp3", files)]
                # println(joinpath(root, i))
                filename = joinpath(root, i)
                x, sr = load_audio(filename, sr=sr)
                # applico normalize e voice detector
                x = normalize_audio(x)
                # da mettere a posto
                # x, _ = speech_detector(x, sr)

                x_length = size(x, 1)
                overlap_length = stft_length - hop_length(x_length, stft_length, n_windows)
                if overlap_length <= 0
                    error("too many windows, try reduce fixed number of windows.")
                end

                afe = extract_audio_features(x, sr, stft_length, overlap_length)'

                # remember that calc n_windows is floored: at least I expect n_windows
                if size(afe, 2) < n_windows
                    error("Something gets wrong with number of windows.")
                else
                    afe = afe[:, 1:n_windows]
                end

                nan_buster(afe, i)
                nan_replacer!(afe)
                
                fileinfo = match(fileregex, i)

                if isempty(X)
                    # initialize dataframe
                    for i in ds_labels
                        X[!, i] = String[]
                    end
                    for i in 1:size(afe, 1)
                        colname = "a$i"
                        X[!, colname] = Vector{Float64}[]
                    end
                end
                push!(X, vcat(mood[fileinfo[1]], fileinfo[1], gender[fileinfo[2]], vcat.(k for k in eachrow(afe))))
            end
        end
    end

    return X
end


#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
n_windows = check_dataset_files(sourcepath, sr, stft_length)

file_set = build_dataframe(sourcepath, sr, stft_length, n_windows)
save_jld2(file_set, string(jld2_path, "emovo_ds_", profile, ".jld2"))

female_set = subset(file_set, :gender => x -> x .== "female")
male_set = subset(file_set, :gender => x -> x .== "male")
save_jld2(female_set, string(jld2_path, "emovo_ds_", profile, "_female.jld2"))
save_jld2(male_set, string(jld2_path, "emovo_ds_", profile, "_male.jld2"))