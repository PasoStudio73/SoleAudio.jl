using DataFrames, JLD2
using Audio911
using StatsBase
using Plots

#--------------------------------------------------------------------------------------#
#                                     parameters                                       #
#--------------------------------------------------------------------------------------#
jld2_path = "/home/paso/Documents/Aclai/experiments_results/speaker_recognition/jld2/"

sr = 8000

sourcepath = [
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_1",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_2",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_3",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_4",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_5",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_6",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_7",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_8",
	"/home/paso/Documents/Aclai/experiments_results/datasets/LibriSpeech/train-clean-360/part_9",
]

#--------------------------------------------------------------------------------------#
#                                    dataset check                                     #
#--------------------------------------------------------------------------------------#
@info "Starting verifying dataset..."

function collect_files(
	source_path::Vector{String},
)
	# initialize id path dataframe
	file_set = DataFrame(id = String[], path = String[], length = Float64[])

	# collect ids
	for single_source_path in source_path
		for (root, _, files) in walkdir(single_source_path)
			for i in files[occursin.(".wav", files).|occursin.(".flac", files).|occursin.(".mp3", files)]
				file_name = vcat(split(i, "-")[1], joinpath(root, i))
				x = load_audio(file=file_name[2], sr=sr)
				# lunghezza audio specificata in secondi
				push!(file_set, vcat(file_name, size(x.data, 1) / sr))
			end
		end
	end

	return file_set
end

#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
# apre la cartella dove sono contenuti i file e crea un unico dataframe
file_set = collect_files(sourcepath)

file_subset = groupby(file_set, :id)

println("total amount of speakers: ", size(file_subset, 1), ", total amount of audio samples: ", size(file_set, 1))
println("audio lengths: 1-mean: ", mean(file_set[!, :length]), " 2-minimum: ", minimum(file_set[!, :length]), " 3-maximum: ", maximum(file_set[!, :length]), " stdev: ", std(file_set[!, :length]))

#--------------------------------------------------------------------------------------#
#                                     plot results                                     #
#--------------------------------------------------------------------------------------#
n_cells = 100
max_length = ceil(Int, maximum(file_set[!, :length]))
step_cell = max_length / n_cells

i = 0
j = 1
data_plot = zeros(Int, n_cells)

while i < max_length
	data_plot[j] = sum(i .< (k[:length] for k in eachrow(file_set)) .< i + step_cell)
	i += step_cell
	j += 1
end

xs = range(0, 30)#maximum(file_set[!, 3]))
# data_plot = sort(file_set[!, 3])
labels = ["length"]
markershapes = [:circle]
markercolors = [:red]

plot(
    # xs,
    data_plot,
    label = labels,
    # shape = markershapes,
    # color = markercolors,
    markersize = 2
)
