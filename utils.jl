# using CSV, DataFrames, JLD2
# using Audio911
# using ModalDecisionTrees
# using SoleDecisionTreeInterface
# using MLJ, Random
# using StatsBase, Catch22
# using CategoricalArrays
# using Plots

# ---------------------------------------------------------------------------- #
#                                  MLJ utils                                   #
# ---------------------------------------------------------------------------- #
function partitioning(X::DataFrame, y::CategoricalArray; train_ratio::Float64=0.8, rng::AbstractRNG=Random.GLOBAL_RNG)
    train, test = partition(eachindex(y), train_ratio, shuffle=true, rng=rng)
    X_train, y_train = X[train, :], y[train]
    X_test, y_test = X[test, :], y[test]
    println("Training set size: ", size(X_train), " - ", length(y_train))
    println("Test set size: ", size(X_test), " - ", length(y_test))
    return X_train, y_train, X_test, y_test
end

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

# function load_jld2(dataset_name::String)
#     # Note: Requires Catch22
#     d = jldopen(string(dataset_name, ".jld2"))
#     df, Y = d["dataframe_validated"]
#     @assert df isa DataFrame
#     close(d)
#     return df, Y
# end

function save_wav_jld2(X::DataFrame, sample_length::Int64, jld2_file::String)
    @info "Save jld2 file..."

    dataframe_validated = (X, sample_length)

    jldsave(jld2_file, true; dataframe_validated)
    println("Dataset: ", jld2_file, " stored.")
end

load_jld2(jld2file::String) = jldopen(jld2file)["jld2_df"]

# ---------------------------------------------------------------------------- #
#                                  csv utils                                   #
# ---------------------------------------------------------------------------- #
function csv2df(csv_path::String; labels::Union{Vector{Symbol}, Vector{AbstractString}, Nothing}=nothing, header::Bool=false, sort::Union{Symbol, Nothing}=nothing)
    csvdf = CSV.read(csv_path, DataFrame, header=header)
    !isnothing(labels) && select!(csvdf, labels)
    return csvdf
end

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

function collect_classes(df::DataFrame, classes_dict::Dict, classes_func::Function; fname::Symbol=:filename, label::Symbol=:label)
    hasproperty(df, fname) || throw(ArgumentError("Column '$fname' does not exist in the DataFrame"))

    labels = map(row -> classes_func(row), eachrow(df))

    if !all(l -> haskey(classes_dict, l), labels)
        missing_keys = setdiff(Set(labels), keys(classes_dict))
        throw(KeyError("The following keys are missing from classes_dict: $missing_keys"))
    end

    DataFrame(Symbol(fname) => df[!, fname], Symbol(label) => map(l -> classes_dict[l], labels))
end

function trimlength_df(df::DataFrame, splitlabel::Symbol, lengthlabel::Symbol, audiolabel::Symbol; sortby::Union{Symbol, Nothing}=nothing, min_length::Int64=0, min_samples::Int64=100, sr::Int64=8000)
    hasproperty(df, splitlabel) ||  throw(ArgumentError("The specified splitlabel, $splitlabel does not exist in the DataFrame."))
    hasproperty(df, lengthlabel) || throw(ArgumentError("The specified lengthlabel, $lengthlabel does not exist in the DataFrame."))
    hasproperty(df, audiolabel) || throw(ArgumentError("The specified audiolabel, $audiolabel does not exist in the DataFrame."))
    
    if !isnothing(sortby) 
        hasproperty(df, sortby) || throw(ArgumentError("The specified sortby, $sortby does not exist in the DataFrame."))
    end

    isnothing(sortby) && begin sortby = lengthlabel end
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
    df[!, lengthlabel] = [length(audio) for audio in df[!, audiolabel]]

    @info "Summary:"
    n_classes = length(grouped_df)
    println("Number of classes: $n_classes")
    println("Samples per class: $(round(Int, nrow(df) / n_classes))")
    sample_length_seconds = actual_length / sr
    println("Sample length: $(round(sample_length_seconds, digits=2)) seconds")

    return df
end

function merge_df_labels!(df::DataFrame, labels::DataFrame; id::Union{Symbol, Nothing}=:filename, id_labels::Symbol=:label)
    size(df, 1) == size(labels, 1) || throw(ArgumentError("DataFrame and Labels must have the same number of rows."))
    all(df[!, id] .== labels[!, id]) || throw(ArgumentError("IDs in DataFrame and Labels do not match row by row."))

    insertcols!(df, id_labels => labels[!, id_labels])
end

sort_df!(df::DataFrame, col::Symbol; rev::Bool=false) = sort!(df, col; rev=rev)

# ---------------------------------------------------------------------------- #
#                                 print utils                                  #
# ---------------------------------------------------------------------------- #
function show_subdf(df::DataFrame, splitlabel::Symbol)
    sub_df = groupby(df, splitlabel)
    for i in sub_df
        println("Sub_df ", i[1,splitlabel], ", total amount of samples: $(nrow(i)).")
    end
end