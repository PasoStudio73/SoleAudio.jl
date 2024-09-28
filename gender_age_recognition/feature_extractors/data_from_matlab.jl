using MAT
using DataFrames, JLD2, CSV

function save_jld2(X::DataFrame, jld2_file::String)
    @info "Save jld2 file..."

    df = X[:, 2:end]
    y = X[:, 1]

    dataframe_validated = (df, y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

ds_type = :gender
# ds_type = :age2bins
# ds_type = :age4bins
# ds_type = :age2split_f
# ds_type = :age2split_m
# ds_type = :age4split_f
# ds_type = :age4split_m

application = :matlab

profile = :full
# profile = :debug
# profile = :gender
# profile = :age

ds_path = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/"
ds_name = string("spcds_", ds_type, "_", application, "_", profile)

if ds_type == :gender
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_gender_matlab_full.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_gender.mat"

elseif ds_type == :age2bins
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2bins_matlab_full.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2bins.mat"
elseif ds_type == :age4bins
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4bins_matlab_full.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4bins.mat"

elseif ds_type == :age2split_f
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2split_matlab_full_female.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2split_female.mat"
elseif ds_type == :age2split_m
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age2split_matlab_full_male.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age2split_male.mat"
elseif ds_type == :age4split_f
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4split_matlab_full_female.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4split_female.mat"
elseif ds_type == :age4split_m
    fileref = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/spcds_age4split_matlab_full_male.csv"
    filename = "/home/riccardopasini/Documents/Aclai/Datasets/Common_voice_ds/6/matlab_exchange/ds_age4split_male.mat"
else
    error("Unknown ds_type: $ds_type.")
end

file = matopen(filename)
from_matlab = read(file, "ads")
total_length = size(from_matlab, 2)

# check for automated division of .mat file
df = DataFrame(CSV.File(fileref))
n_samples = size(df, 1)

n_features = Int(total_length / n_samples)

# create dataframe for audio features storing
X = DataFrame()
X[!, ds_type] = String[]
for i in 1:n_features
    colname = "a$i"
    X[!, colname] = Vector{Float64}[]
end

for i in 1:n_samples
    first_index = (i - 1) * n_features + 1
    push!(X, vcat(string(df[i, :id]), vcat.(k for k in eachcol(from_matlab[:, first_index:first_index+n_features-1]))))
end

save_jld2(X, string(ds_path, ds_name, ".jld2"))