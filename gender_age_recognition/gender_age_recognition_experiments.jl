using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using StatsBase, Catch22

include("../utils.jl")
include("gender_age_recognition.jl")

@info "Starting..."

ds_path = "/home/paso/Documents/Aclai/Datasets/gender_age_recognition/6/"

# ds_csv = "/home/paso/Documents/Aclai/Datasets/gender_age_recognition/3/common_voice_ds_3.csv"
ds_csv = "/home/paso/Documents/Aclai/Datasets/gender_age_recognition/6/common_voice_ds_6.csv"

result_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/gender_age/"

#--------------------------------------------------------------------------------------#
#                                       settings                                       #
#--------------------------------------------------------------------------------------#
ds_type = :gender
# ds_type = :age2bins
# ds_type = :age4bins
# ds_type = :age2split
# ds_type = :age4split

propositional = false # modal
# propositional = true

n_samples = 2000
n_samples = 1000
seed = 11

# debug = false
# debug = true

#--------------------------------------------------------------------------------------#
#                                     initial setup                                    #
#--------------------------------------------------------------------------------------#
ds_name = string("spcds_", ds_type, "_", propositional ? "prop" : "modal")

ds_params = DsParams(ds_path, ds_csv, ds_type, ds_name, n_samples, seed, result_path, propositional, catch9)

# if !debug
    gender_age_recognition(ds_params)
# else
#     X = gender_age_recognition(ds_params)
# end

@info "Done."