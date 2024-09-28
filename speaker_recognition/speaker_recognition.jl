using DataFrames, JLD2, CSV

#--------------------------------------------------------------------------------------#
#                                     parameters                                       #
#--------------------------------------------------------------------------------------#
sigma_threshold = 0.95
max_valid_dist = 3.0
ddd = 1.0

#--------------------------------------------------------------------------------------#
#                                anonymous functions                                   #
#--------------------------------------------------------------------------------------#
# 3 sigma
is_3sigma_feat = (feat, centroid, std_dv) -> centroid - 3 * std_dv .<= feat .<= centroid + 3 * std_dv
# euclidean distance
calc_distance = (feat, centroid) -> sqrt(sum((feat .- centroid) .^ 2))
# count valid feats
count_valid = (sigma) -> length(filter(!!, sigma))
# check if sample has enough valid features
check_threshold = (f, total_feats) -> f / total_feats >= sigma_threshold ? true : false
# narrow feat
is_narrow_feat = (feat, centroid, std_dv) -> centroid - std_dv <= feat <= centroid + std_dv

#--------------------------------------------------------------------------------------#
#                                 datatype and structs                                 #
#--------------------------------------------------------------------------------------#
if !@isdefined(TestSample)
	mutable struct TestSample
		id::String
		feats::AbstractVector{Float64}
		sigma::Vector{Tuple{String, Vector{Bool}}}
		distances::Vector{Tuple{String, Float64}}
		status::Symbol
		guessed_ids::Vector{Tuple{String, Float64}}

		function TestSample(
			id::String,
			feats::Vector{Float64},
			train_df::DataFrame,
			train_y::Vector{String},
		)
			# train_id = Dict(j .=> train_y[j] for j in 1:length(train_y))
			valid_feats = Tuple{String, Vector{Bool}}[]
			distances = Tuple{String, Float64}[]

			for j in axes(train_df, 1)
				push!(valid_feats, (train_y[j], map(is_3sigma_feat, feats, train_df[j, :centroids], train_df[j, :std_dv])))
				push!(distances, (train_y[j], calc_distance(feats, train_df[j, :centroids])))
			end
			new(id, feats, valid_feats, distances, :unknown, []) # :known, :overlap, :unknown
		end
	end
end

#--------------------------------------------------------------------------------------#
#                                    build test set                                    #
#--------------------------------------------------------------------------------------#
function build_test_set(
	train_df::DataFrame,
	train_y::Vector{String},
	test_df::DataFrame,
	test_y::Vector{String},
)
	test_set = TestSample[]
	for i in eachindex(test_y)
		# crea l'oggetto
		test_sample = TestSample(
			test_y[i],# id
			test_df[i, 1],# features
			train_df,# train set per calcoli
			train_y,# ids train set
			# test_df[i, 2],	# se realmente conosciuto
		)
		push!(test_set, test_sample)
	end

	return test_set
end

#--------------------------------------------------------------------------------------#
#                                  speaker recognition                                 #
#--------------------------------------------------------------------------------------#
function check_sigma(
	test_set::Vector{TestSample},
)
	for i in test_set
		for j in eachindex(i.sigma)
			_, sigma = i.sigma[j]
			id, distances = i.distances[j]
			f = count_valid(sigma)

			if check_threshold(f, length(i.feats)) && i.status == :unknown
				i.status = :known
				push!(i.guessed_ids, (id, distances))

			elseif check_threshold(f, length(i.feats)) && i.status == :known
				i.status = :overlap
				push!(i.guessed_ids, (id, distances))

			elseif check_threshold(f, length(i.feats)) && i.status == :overlap
				push!(i.guessed_ids, (id, distances))
			end
		end
	end
end

function check_overlap(
	test_set::Vector{TestSample},
)
	for i in test_set
		if i.status == :overlap
			dist = sort(i.guessed_ids, by = x -> x[2])
			id, best_dist = dist[1]
			if best_dist < max_valid_dist
				i.status = :known
				i.guessed_ids = [(id, best_dist)]
			else
				i.status = :unknown
			end
		end
	end
end

function check_distances(
	test_set::Vector{TestSample},
)
	for i in test_set
		if i.status == :unknown
			dist = sort(i.distances, by = x -> x[2])
			id, best_dist = dist[1]
			if best_dist < ddd
				i.status = :known
				i.guessed_ids = [(id, best_dist)]
			end
		end
	end
end

function calc_accuracy(
	test_set::Vector{TestSample},
	train_y::Vector{String},
)
	ntotal = size(test_set, 1)
	ncorrects = 0
	for i in test_set
		if (i.status == :unknown && !(i.id in train_y))
			ncorrects += 1
		elseif (i.status == :known)
			id, a = i.guessed_ids[1]
			if i.id == id
				ncorrects += 1
			end
		end
	end
	return ncorrects / ntotal
end