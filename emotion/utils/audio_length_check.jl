using DataFrames
using Audio911
# using JLD2
using StatsBase
# using Plots

#--------------------------------------------------------------------------------------#
#                                     parameters                                       #
#--------------------------------------------------------------------------------------#
# jld2_path = "/home/paso/Documents/Aclai/experiments_results/speaker_recognition/jld2/"

sr = 8000

sourcepath = [
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f1",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f2",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/f3",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m1",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m2",
    "/home/paso/Documents/Aclai/Datasets/emotion_recognition/datasets/Emovo_dataset/EMOVO_dataset/m3",
]

#--------------------------------------------------------------------------------------#
#                                    dataset check                                     #
#--------------------------------------------------------------------------------------#
@info "Starting verifying dataset..."

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
                # applico normalize e voice detector
                x = normalize_audio(x)
                x = speech_detector(x, sr)
                # lunghezza audio specificata in samples
                push!(file_set, vcat(file_name, size(x, 1)))
            end
        end
    end

    return file_set
end

#--------------------------------------------------------------------------------------#
#                                       utility                                        #
#--------------------------------------------------------------------------------------#
function calc_n_windows(
    x_min::Int64,
    x_max::Int64,
)
    num_hops = (x, window_length, hop_length) -> floor(Int, (x - window_length) / hop_length) + 1
    hop_length = (x, window_length, num_hops) -> floor(Int, (x - window_length) / num_hops) + 1

    stft_length = 256

    best_max = num_hops(x_max, stft_length, floor(Int, stft_length / 2))
    println("finestre per coprire al meglio il sample più lungo: ", best_max)

    hop_min = hop_length(x_min, stft_length, best_max)
    println("avanzamento nel sample più corto: ", hop_min)
    println("equivalente al: ", hop_min * 100 / stft_length, "%")

    # matlab_window_length::Int64 = round(Int, 0.03 * sr)
    matlab_hop_length::Int64 = round(Int, stft_length - 0.025 * sr)

    best_min = num_hops(x_min, stft_length, matlab_hop_length)
    println("finestre per coprire al meglio il sample più corto: ", best_min)

    hop_max = hop_length(x_max, stft_length, best_min)
    println("avanzamento nel sample più lungo: ", hop_max)
    println("equivalente al: ", hop_max * 100 / stft_length, "%")

    n_windows = floor(Int, (best_max + best_min) / 2)
    println("numero finestre proposto: ", n_windows)

    return n_windows
end

#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
# apre la cartella dove sono contenuti i file e crea un unico dataframe
file_set = collect_files_length(sourcepath, sr)

# file_subset = groupby(file_set, :id)

println("total amount of audio samples: ", size(file_set, 1))
x_mean = floor(Int, mean(file_set[!, :length]))
x_median = floor(Int, median(file_set[!, :length]))
x_min = minimum(file_set[!, :length])
x_max = maximum(file_set[!, :length])

println("audio lengths: mean: ", x_mean, ", median: ", x_median)
println("minimum: ", x_min, " maximum: ", x_max)

# for i in eachrow(file_set)
# 	if i[:length] == 0
# 		println(i[:id])
# 	end
# end

n_windows = calc_n_windows(x_min, x_max)



#--------------------------------------------------------------------------------------#
#                                     plot results                                     #
#--------------------------------------------------------------------------------------#
# n_cells = 100
# max_length = ceil(Int, maximum(file_set[!, :length]))
# step_cell = max_length / n_cells

# i = 0
# j = 1
# data_plot = zeros(Int, n_cells)

# while i < max_length
# 	data_plot[j] = sum(i .< (k[:length] for k in eachrow(file_set)) .< i + step_cell)
# 	i += step_cell
# 	j += 1
# end

# xs = range(0, 30)#maximum(file_set[!, 3]))
# # data_plot = sort(file_set[!, 3])
# labels = ["length"]
# markershapes = [:circle]
# markercolors = [:red]

# plot(
#     # xs,
#     data_plot,
#     label = labels,
#     # shape = markershapes,
#     # color = markercolors,
#     markersize = 2
# )
