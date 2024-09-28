using CSV, DataFrames, JLD2
using StatsBase, Catch22
using Audio911

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
catch9 = [
    maximum,
    minimum,
    StatsBase.mean,
    median,
    std,
    Catch22.SB_BinaryStats_mean_longstretch1,
    Catch22.SB_BinaryStats_diff_longstretch0,
    Catch22.SB_MotifThree_quantile_hh,
    Catch22.SB_TransitionMatrix_3ac_sumdiagcov,
]

# ---------------------------------------------------------------------------- #
#                                  jld2 utils                                  #
# ---------------------------------------------------------------------------- #
function save_jld2(X::DataFrame, jld2_file::String; labels::Int)
    @info "Save jld2 file..."

    df = X[:, labels+1:end]
    y = Array(X[:, 1:labels])

    dataframe_validated = (df, y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function save_jld2(X::DataFrame, Y::AbstractVector{<:AbstractString}, jld2_file::String)
    @info "Save jld2 file..."

    dataframe_validated = (X, Y)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
    println("Features: ", size(df, 2), ", rows: ", size(df, 1), ".")
end

function save_jld2(X::AbstractVector{Matrix{Float64}}, args...)
    save_jld2(DataFrame((vec(m) for m in X), :auto), args...)
end

function load_jld2(dataset_name::String)
    # Note: Requires Catch22
    d = jldopen(string(dataset_name, ".jld2"))
    df, Y = d["dataframe_validated"]
    @assert df isa DataFrame
    close(d)
    return df, Y
end

function save_wav_jld2(X::DataFrame, sample_length::Int64, jld2_file::String)
    @info "Save jld2 file..."

    dataframe_validated = (X, sample_length)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
end

function load_wav_jld2(jld2_file::String)
    d = jldopen(jld2_file)
    return d["dataframe_validated"]
end

# ---------------------------------------------------------------------------- #
#                                 audio utils                                  #
# ---------------------------------------------------------------------------- #
"""
Calculate the length of audio vectors in a specified column of a DataFrame and add a new column with these lengths.

Arguments:
- `df::DataFrame`: The input DataFrame containing the audio data.
- `audiolabel::Symbol`: The symbol representing the column name that contains the audio vectors.

Returns:
- `DataFrame`: The modified DataFrame with a new column `:audio_length` containing the lengths of the audio vectors.
"""
function get_audio_length!(df::DataFrame, audiolabel::Symbol, lengthlabel::Symbol)
    insertcols!(df, lengthlabel => size.(df[:, audiolabel], 1))
end

# function _collect_audio_from_folder!(df::DataFrame, path::String, sr::Int64; norm::Bool=true, spc_detect::Bool=false)
#     # collect files
#     for (root, _, files) in walkdir(path)
#         for file in filter(f -> any(occursin.([".wav", ".flac", ".mp3"], f)), files)
#             x = load_audio(joinpath(root, file); sr=sr, norm=norm, spc_detect=spc_detect)
#             spc_detect && (x.data = speech_detector(x))
#             push!(df, hcat(split(file, ".")[1], size(x.data, 1), [x.data]))
#         end
#     end
# end

# function collect_audio_from_folder(path::String; sr::Int64, kwargs...)
#     @info "Collect files..."
#     # initialize id path dataframe
#     df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
#     _collect_audio_from_folder!(df, path, sr; kwargs...)
#     return df
# end

# function collect_audio_from_folder(path::AbstractVector{String}; sr::Int64, kwargs...)
#     @info "Collect files..."
#     df = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])

#     for i in path
#         _collect_audio_from_folder!(df, i, sr; kwargs...)
#     end

#     return df
# end

function audio_features(x::AbstractVector{Float64}, audioparams::NamedTuple; spc_detect::Bool=false)
    Audio911.afe(x; audioparams...)
end

function audio_features(df::DataFrame, audioparams::NamedTuple; kwargs...)
    @info "Extracting audio features..."
    insertcols!(df, ncol(df)+1, :afe => fill(missing, nrow(df)))
    df[!, :afe] = map(x -> audio_features(x, audioparams), df[!, :audio]; kwargs...)
end

# ---------------------------------------------------------------------------- #
#                                  csv utils                                   #
# ---------------------------------------------------------------------------- #
"""
Reads a CSV file and returns a DataFrame.

Arguments:
- `csv_path::String`: The path to the CSV file.
- `labels::Union{Vector{Symbol}, Vector{AbstractString}, Nothing}=nothing`: The column labels to select from the CSV file. If `nothing`, all columns are selected.
- `header::Bool=false`: Whether the CSV file has a header row.
- `sort::Union{Symbol, Nothing}=nothing`: The column to sort the DataFrame by.

Returns:
A DataFrame containing the data from the CSV file.
"""
function csv2df(csv_path::String; labels::Union{Vector{Symbol}, Vector{AbstractString}, Nothing}=nothing, header::Bool=false, sort::Union{Symbol, Nothing}=nothing)
    csvdf = CSV.read(csv_path, DataFrame, header=header)
    !isnothing(labels) && select!(csvdf, labels)
    return csvdf
end

"""
Merges a DataFrame with data from a CSV file.

Arguments:
- `df::DataFrame`: The DataFrame to merge the CSV data into.
- `csvdf::DataFrame`: The DataFrame read from the CSV file.
- `id_df::Union{Symbol, Nothing}=nothing`: The column in `df` to use as the identifier for the merge.
- `id_csv::Union{Symbol, Nothing}=nothing`: The column in `csvdf` to use as the identifier for the merge.

The function first sorts the DataFrames by the specified identifier columns if provided. It then inserts the columns from `csvdf` into `df`, excluding the identifier column from `csvdf`.
"""
function merge_df_csv!(df::DataFrame, csvdf::DataFrame; id_df::Union{Symbol, Nothing}=nothing, id_csv::Union{Symbol, Nothing}=nothing)
    size(df, 1) == size(csvdf, 1) || throw(ArgumentError("DataFrame and Csv must have the same number of rows."))

    if !isnothing(id_df) && !isnothing(id_csv)
        sort!(df, id_df)
        sort!(csvdf, id_csv)
    end
    labels = filter(col -> col != id_csv, names(csvdf))
    insertcols!(df, [col => csvdf[!, col] for col in labels]...)
end

# kept for compatibility
function csv2dict(df::DataFrame)
    Dict(zip(string.(df[:, 1]), string.(df[:, 2])))
end
# kept for compatibility
function csv2dict(file::String)
    csv2dict(CSV.read(file, DataFrame, header=false))
end

# ---------------------------------------------------------------------------- #
#                               dataframe utils                                #
# ---------------------------------------------------------------------------- #
function group_df(grouped_df::GroupedDataFrame{DataFrame}, label::Symbol)
    @info "Balancing classes..."
    nsamples = minimum(nrow, grouped_df)

    combine(grouped_df) do i
        first(i, nsamples)
    end
end

function group_df(df::DataFrame, splitlabel::Symbol; sortby::Union{Symbol, Nothing}=nothing)
    !isnothing(sortby) && sort!(df, sortby, rev=true)
    grouped_df = groupby(df, splitlabel)
    group_df(grouped_df, splitlabel)
end

function group_df(df::DataFrame, splitlabel::Symbol, keep_only::AbstractVector{String})
    df = filter(row -> row[splitlabel] in keep_only, df)
    sub_df = groupby(df, splitlabel)
    group_df(sub_df, splitlabel)
end

function trimlength_df(df::DataFrame, splitlabel::Symbol, lengthlabel::Symbol, audiolabel::Symbol; sortby::Union{Symbol, Nothing}=nothing, min_length::Int64=0, min_samples::Int64=100)
    string(splitlabel) in names(df) ||  throw(ArgumentError("The specified splitlabel, $splitlabel does not exist in the DataFrame."))
    string(lengthlabel) in names(df) || throw(ArgumentError("The specified lengthlabel, $lengthlabel does not exist in the DataFrame."))
    string(audiolabel) in names(df) || throw(ArgumentError("The specified audiolabel, $audiolabel does not exist in the DataFrame."))
    if !isnothing(sortby) 
        string(sortby) in names(df) || throw(ArgumentError("The specified sortby, $sortby does not exist in the DataFrame."))
    end

    isnothing(sortby) && (sortby = lengthlabel)
    sort!(df, sortby, rev=true)

    grouped_df = groupby(df, splitlabel)

    first_lengths = [group[1, lengthlabel] for group in grouped_df]
    all(group -> nrow(group) ≥ min_samples, grouped_df) || throw(ArgumentError("Some groups have less than $min_samples samples. Try reducing min_samples."))
    all(length -> length ≥ min_length, first_lengths) || throw(ArgumentError("Some groups have first sample length less than $min_length. Try reducing min_length."))

    min_length_ids = [
        let last_valid = findlast(≥(min_length), group[!, lengthlabel])
            isnothing(last_valid) ? nrow(group) : last_valid
        end for group in grouped_df]
    min_length_id = minimum(min_length_ids)

    min_length_id ≥ min_samples || throw(ArgumentError("The number of samples ($(min_length_id)) meeting the minimum length requirement is less than the specified minimum sample count ($(min_samples)). Consider reducing the min_samples parameter or adjusting the min_length value."))
    
    actual_length = minimum(group[min_length_id, lengthlabel] for group in grouped_df)
    df = combine(grouped_df) do i
        first(i, min_length_id)
    end
    df[!, audiolabel] = [audio[max(1, length(audio) - actual_length + 1):end] for audio in df[!, audiolabel]]

    return df
end

function read_filenames(df::DataFrame, classes_func::Function, classes_dict::Dict; id::Union{Symbol, Nothing}=nothing, label::Symbol=:label)
    DataFrame(label = map(row -> classes_dict[classes_func(row)], eachrow(df)))
end

function show_subdf(df::DataFrame, splitlabel::Symbol)
    sub_df = groupby(df, splitlabel)
    for i in sub_df
        println("Sub_df ", i[1,splitlabel], ", total amount of samples: $(nrow(i)).")
    end
end