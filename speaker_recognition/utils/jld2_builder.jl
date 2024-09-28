using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Audio911
using DataFrames, JLD2
using Random, StatsBase

include("../../afe.jl")
include("../../utils.jl")

#--------------------------------------------------------------------------------------#
#                                    data structures                                   #
#--------------------------------------------------------------------------------------#
Base.@kwdef struct DatasetParameters
	train_samples::Int
	files_per_index::Int
	test_samples::Int
	id_ratio::Int
	source_ds::Symbol
	sourcepath::Vector{String}
	sr::Int64
	stft_length::Int64
	freq_range::Tuple{Int64, Int64}
end

# Base.@kwdef struct AudioParameters
# 	sr::Int64
# 	profile::Symbol
# 	stft_length::Int64
# 	freq_range::Tuple{Int64, Int64}
# 	mel_bands::Int64
# 	mfcc_coeffs::Int64
# end

#--------------------------------------------------------------------------------------#
#                             parameters multiple speakers                             #
#--------------------------------------------------------------------------------------#
seeds = [1, 4, 9, 11, 23, 34, 43, 46, 58, 65]

# amount of speakers in train set
train_samples = [1, 2, 4, 7, 10, 15, 20]

# samples per single speaker in training
files_per_index = 20

# numero di sample da prendere per ogni id che compone il test set
test_samples = 40
# rapporto tra id conosciuti e id sconosciuti: 
# se = 1 allora avrò x parlanti conosciuti e x parlanti sconosciuti,
# se = Inf allora userò tutti gli id disponibili come possibili parlanti sconosciuti
id_ratio = 2

reduction_type = :mean
# reduction_type = :median

# norm_type = :unit_range
norm_type = :x_score

#--------------------------------------------------------------------------------------#
#                                additional parameters                                 #
#--------------------------------------------------------------------------------------#
jld2_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/speaker_recognition/"

#--------------------------------------------------------------------------------------#
#                                    dataset utils                                     #
#--------------------------------------------------------------------------------------#
ts_reduce = Dict{Symbol, Function}(
	:mean => x::DataFrame -> dataframe2array(mean.(x)),
	:maximum => x::DataFrame -> dataframe2array(maximum.(x)),
	:median => x::DataFrame -> dataframe2array(median.(x)),
	:mode => x::DataFrame -> dataframe2array(mode.(x)),
)

cs_reduce = Dict{Symbol, Function}(
	:mean => x::AbstractArray{Float64} -> mean(x, dims = 1),
	:maximum => x::AbstractArray{Float64} -> maximum(x, dims = 1),
	:median => x::AbstractArray{Float64} -> median(x, dims = 1),
	:mode => x::AbstractArray{Float64} -> mode(x, dims = 1),
)

ts_norm = Dict{Symbol, Function}(
	:unit_range => x::AbstractArray -> fit(UnitRangeTransform, x, dims = 1),
	:x_score => x::AbstractArray -> fit(ZScoreTransform, x, dims = 1),
)

function dataframe2array(df::AbstractDataFrame)
	dataset = [
		begin
			instance = begin
				cat(collect(row)...; dims = ndims(row[1]) + 1)
			end
			instance
		end for row in eachrow(df)
	]
	stack(dataset, dims = 1)
end

function array2sets(
	X::AbstractArray{Float64},
	y::Vector{String},
	n_samples::Int64,
)
	samplesets = Vector{Vector{Float64}}[]
	Y = String[]

	for i in 1:n_samples:size(X, 1)
		push!(samplesets, [X[j, :] for j in i:i+n_samples-1])
		push!(Y, y[i])
	end

	return samplesets, Y
end

#--------------------------------------------------------------------------------------#
#                              audio features extraction                               #
#--------------------------------------------------------------------------------------#
function audio_features_collect(filepath::String, dp::DatasetParameters)
	# becomes true if sample is valid (length > stft_length after speech_detector)
	valid = false
	# load wav file
	x = load_audio(file=filepath, sr=dp.sr, norm=true)
	x = speech_detector(audio=x)

	if size(x.data, 1) > dp.stft_length
		# audio feature extraction (Matrix{Float64} -> (rows:time, cols:features))
		return audio911_extractor(
			x.data,
			sr=dp.sr,
			stft_length=dp.stft_length,
			freq_range=dp.freq_range,
		)
	end
end

#--------------------------------------------------------------------------------------#
#                                  dataset building                                    #
#--------------------------------------------------------------------------------------#
function setup_dataset(
	dp::DatasetParameters,
	seed::Int64,
	reduction_type::Symbol,
	norm_type::Symbol,
)
	@info "Starting building dataset..."

	function collect_files(
		source_ds::Symbol,
		source_path::Vector{String},
	)
		function id(name::String, source_dataset::Symbol)
			if source_dataset == :googlespeech
				split(name, "_")[1]
			elseif source_dataset == :librispeech
				split(name, "-")[1]
			else
				error("Unknown dataset source $source_dataset.")
			end
		end
		# initialize id path dataframe
		file_set = DataFrame(id = String[], path = String[], test = Bool[])

		# collect ids
		for single_source_path in source_path
			for (root, _, files) in walkdir(single_source_path)
				for i in files[occursin.(".wav", files).|occursin.(".flac", files).|occursin.(".mp3", files)]
					push!(file_set, vcat(id(i, source_ds), joinpath(root, i), false))
				end
			end
		end

		return file_set
	end

	function groupby_ids(file_set::DataFrame, train_samples::Int, id_ratio::Union{Int64, Float64})
		file_subset = groupby(file_set, :id)
		if length(file_subset) < train_samples
			@warn "not enough samples, use replace=true"
			choosen_indexes = sample(rng, 1:length(file_subset), train_samples, replace = true)
		else
			choosen_indexes = sample(rng, 1:length(file_subset), train_samples, replace = false)
		end

		# apply id ratio for unknown ids
		ts = train_samples * id_ratio
		if id_ratio == Inf || ts > length(file_subset[setdiff(collect(range(1, length(file_subset))), choosen_indexes)])
			unknown_indexes = setdiff(collect(range(1, length(file_subset))), choosen_indexes)
			all_indexes = true
		elseif length(file_subset[setdiff(collect(range(1, length(file_subset))), choosen_indexes)]) < train_samples
			@warn "not enough samples, use replace=true"
			unknown_indexes = sample(rng, setdiff(collect(range(1, length(file_subset))), choosen_indexes), ts, replace = true)
			all_indexes = false
		else
			unknown_indexes = sample(rng, setdiff(collect(range(1, length(file_subset))), choosen_indexes), ts, replace = false)
			all_indexes = false
		end

		return file_subset, choosen_indexes, unknown_indexes, all_indexes
	end

	function compile(
		X::DataFrame,
		choosen_files,
		file_subset,
		i;
		known::Bool = false,
	)
		for j in choosen_files
			afe = audio_features_collect(file_subset[i].path[j], dp)

			# println(file_subset[i].path[j])

			# if valid
			if isempty(X)
				# initialize dataframe
				X[!, "id"] = String[]
				for i in 1:size(afe, 2)
					colname = "a$i"
					X[!, colname] = Vector{Float64}[]
				end
				X[!, "known"] = Bool[]
			end

			push!(X, vcat(file_subset[i].id[j], vcat.(k for k in eachcol(afe)), known))

		end
	end

	# initialize random seed
	Random.seed!(seed)
	rng = MersenneTwister(seed)

	# apre la cartella dove sono contenuti i file e crea un unico dataframe
	file_set = collect_files(dp.source_ds, dp.sourcepath)

	# groupby ids and pick them randomly
	# sceglie per quanti speaker avrò, i loro subset a caso (sceglie gli speaker)
	file_subset, choosen_indexes, unknown_indexes, all_indexes = groupby_ids(file_set, dp.train_samples, dp.id_ratio)

	train_set = DataFrame()
	test_set = DataFrame()

	println("choosen indexes: ", choosen_indexes, ", unknown indexes: ", all_indexes ? "all availables" : unknown_indexes)

	for i in choosen_indexes
		# per ogni indice scelgo dei sample a caso
		# la quantità necessaria è di files_per_index + test_samples
		# ovvero i sample che compongono il train_set più il sample da testare
		if size(file_subset[i], 1) < (dp.files_per_index + dp.test_samples)
			@warn "not enough samples, use replace=true"
			choosen_files = sample(rng, 1:size(file_subset[i], 1), dp.files_per_index + dp.test_samples, replace = true)
		else
			choosen_files = sample(rng, 1:size(file_subset[i], 1), dp.files_per_index + dp.test_samples, replace = false)
		end

		compile(train_set, choosen_files[1:dp.files_per_index], file_subset, i)
		compile(test_set, choosen_files[dp.files_per_index+1:end], file_subset, i, known = true)
	end

	# adding unknown speakers
	unknown_samples = dp.train_samples * dp.test_samples

	if all_indexes
		for i in 1:unknown_samples
			choosen_id = sample(rng, unknown_indexes, 1)
			choosen_file = sample(rng, 1:size(file_subset[choosen_id...], 1), 1)
			compile(test_set, choosen_file, file_subset, choosen_id..., known = false)
		end
	else
		n = ceil(Int, unknown_samples / length(unknown_indexes))
		n_max = size(test_set, 1) + unknown_samples

		for i in unknown_indexes
			if size(file_subset[i], 1) < n
				@warn "not enough samples, use replace=true"
				choosen_files = sample(rng, 1:size(file_subset[i], 1), n, replace = true)
			else
				choosen_files = sample(rng, 1:size(file_subset[i], 1), n, replace = false)
			end
			compile(test_set, choosen_files, file_subset, i, known = false)
		end
		test_set = test_set[1:n_max, :]
	end

	# trasformo i DataFrame di audio features in array di matrici (n_samples x audio_features)
	# per poi normalizzarle
	train_ids = train_set.id
	test_ids = test_set.id
	test_knowns = test_set.known

	# reducing time series
	trainX = ts_reduce[reduction_type](train_set[!, 2:end-1])
	testX = ts_reduce[reduction_type](test_set[!, 2:end-1])

	# features normalization
	t = ts_norm[norm_type](vcat(trainX, testX))

	trainX = StatsBase.transform(t, trainX)
	testX = StatsBase.transform(t, testX)

	# initialize id path dataframe
	train_df = DataFrame(id = String[], centroids = Vector{Float64}[], std_dv = Vector{Float64}[])
	test_df = DataFrame(id = String[], features = Vector{Float64}[], known = Bool[])

	sets, sets_id = array2sets(trainX, train_ids, files_per_index)

	for i in axes(sets, 1)
		centroids = cs_reduce[reduction_type](reduce(hcat, sets[i])')
		std_dv = std(reduce(hcat, sets[i])', dims = 1)

		push!(train_df, (sets_id[i], vec(centroids), vec(std_dv)))
	end

	for i in axes(test_ids, 1)
		push!(test_df, (test_ids[i], testX[i, :], test_knowns[i]))
	end

	return train_df, test_df
end

#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
# cicla tra i seed e il numero di train samples
for i in seeds
	for j in train_samples
		dataset_parameters = DatasetParameters(
			train_samples=j,
			files_per_index=files_per_index,
			test_samples=test_samples,
			id_ratio=id_ratio,
			# dataset:
			# LibriSpeech ASR Corpus
			# http://www.openslr.org/12
			source_ds=:librispeech,
			sourcepath=[
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_1",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_2",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_3",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_4",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_5",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_6",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_7",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_8",
				"/home/paso/Documents/Aclai/Datasets/speaker_recognition/LibriSpeech/train-clean-360/part_9",
			],
			sr=8000,
			stft_length=1024,
			freq_range=(0, 4000)
		)

		train_df, test_df = setup_dataset(
			dataset_parameters,
			i,
			reduction_type,
			norm_type,
		)

		save_jld2(train_df, string(jld2_path, "srTrain_", i, "_", j, ".jld2"))
		save_jld2(test_df, string(jld2_path, "srTest_", i, "_", j, ".jld2"))

		## debug
		# save_jld2(train_df, string(jld2_path, "srTrain_", debug, "_", j, ".jld2"))
		# save_jld2(test_df, string(jld2_path, "srTest_", debug, "_", j, ".jld2"))
	end
end
