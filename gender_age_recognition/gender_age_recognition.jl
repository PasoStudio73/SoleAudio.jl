using Audio911, Catch22
using Random, StatsBase
using DataFrames, CSV, JLD2

include("../afe.jl")

#--------------------------------------------------------------------------------------#
#                                    data structures                                   #
#--------------------------------------------------------------------------------------#
struct DsParams
    path::String
    csv::String
    type::Symbol
    name::String
    n_samples::Int64
    seed::Int64
    result_path::String
    propositional::Bool
    catch9::AbstractVector{Function}
end

#--------------------------------------------------------------------------------------#
#                                         utils                                        #
#--------------------------------------------------------------------------------------#
function afe!(X::DataFrame, ds_params::DsParams, ds_type::Symbol, i, choosen_indexes)
    for index in choosen_indexes
        # println(string(ds_params.path, "Wavfiles/", i[index, :path]))
        # println("***************")
        x_features = audio911_extractor(string(ds_params.path, "Wavfiles/", i[index, :path]))

        if ds_params.propositional
            x_features = hcat(collect(map(func -> func(i), ds_params.catch9) for i in eachcol(x_features))...)

            # feats = Float64[]
            # for i in eachcol(x_features)
            #     push!(feats, map(func -> func(i), ds_params.catch9)...)
            # end
            # x_features = feats
        end

        nan_replacer!(x_features)

        if isempty(X)
            # initialize dataframe
            X[!, ds_type] = String[]
            if ds_params.propositional
                for i in 1:length(x_features)
                    colname = "a$i"
                    X[!, colname] = Float64[]
                end
            else
                for i in 1:size(x_features, 2)
                    colname = "a$i"
                    X[!, colname] = Vector{Float64}[]
                end
            end
        end

        if ds_params.propositional
            push!(X, vcat(i[index, ds_type], x_features...))
        else
            push!(X, vcat(i[index, ds_type], vcat.(k for k in eachrow(x_features'))))
        end
    end
end

#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
function gender_age_recognition(ds_params::DsParams)

    # initialize random seed
    Random.seed!(ds_params.seed)
    rng = MersenneTwister(ds_params.seed)

    # load and subdivde csv
    df = DataFrame(CSV.File(ds_params.csv))

    if ds_params.type == :gender
        ds_type = :gender

    elseif ds_params.type == :age2bins
        replacement_dict = Dict([
            "teens" => "25", "twenties" => "25",
            "thirties" => "25", "fourties" => "skip",
            "fifties" => "skip", "sixties" => "70",
            "seventies" => "70", "eighties" => "70", "nineties" => "70",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        df = filter(row -> !(row.age == "skip"), df)
        ds_type = :age

    elseif ds_params.type == :age4bins
        replacement_dict = Dict([
            "teens" => "20", "twenties" => "20",
            "thirties" => "40", "fourties" => "40",
            "fifties" => "60", "sixties" => "60",
            "seventies" => "80", "eighties" => "80", "nineties" => "80",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        ds_type = :age

    elseif ds_params.type == :age8bins
        replacement_dict = Dict([
            "teens" => "15", "twenties" => "25",
            "thirties" => "35", "fourties" => "45",
            "fifties" => "55", "sixties" => "65",
            "seventies" => "75", "eighties" => "85", "nineties" => "85",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        ds_type = :age

    elseif ds_params.type == :age2split
        replacement_dict = Dict([
            "teens" => "25", "twenties" => "25",
            "thirties" => "25", "fourties" => "skip",
            "fifties" => "skip", "sixties" => "70",
            "seventies" => "70", "eighties" => "70", "nineties" => "70",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        df = filter(row -> !(row.age == "skip"), df)
        ds_type = [:age, :gender]

    elseif ds_params.type == :age4split
        replacement_dict = Dict([
            "teens" => "20", "twenties" => "20",
            "thirties" => "40", "fourties" => "40",
            "fifties" => "60", "sixties" => "60",
            "seventies" => "80", "eighties" => "80", "nineties" => "80",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        ds_type = [:age, :gender]

    elseif ds_params.type == :age8split
        replacement_dict = Dict([
            "teens" => "15", "twenties" => "25",
            "thirties" => "35", "fourties" => "45",
            "fifties" => "55", "sixties" => "65",
            "seventies" => "75", "eighties" => "85", "nineties" => "85",
        ])
        df.age .= replace.(df.age, replacement_dict...)
        ds_type = [:age, :gender]
    else
        error("Unknown dataset type $(ds_params.type)")
    end

    sub_df = DataFrames.groupby(df, ds_type)

    @info "Dataset compiling..."

    # age, female, male split
    if ds_type == [:age, :gender]
        # if debug
        #     error("debug not implemented for splitted datasets.")
        # end

        n_samples = Int(round(ds_params.n_samples / 2))

        # create dataframe for audio features storing
        X = DataFrame()

        ## randomly pick a sample, save reference to a csv_dest and store wavfile
        new_df = subset(sub_df, :gender => x -> x .== "female")
        new_sub_df = DataFrames.groupby(new_df, ds_type[1])

        for i in new_sub_df
            if nrow(i) < n_samples
                @warn "not enough samples, use replace=true"
                choosen_indexes = sample(rng, 1:nrow(i), n_samples, replace=true)
            else
                choosen_indexes = sample(rng, 1:nrow(i), n_samples, replace=false)
            end

            afe!(X, ds_params, ds_type[1], i, choosen_indexes)
        end

        save_jld2(X, string(ds_params.result_path, ds_params.name, "_female.jld2"))

        # create dataframe for audio features storing
        X = DataFrame()

        new_df = subset(sub_df, :gender => x -> x .== "male")
        new_sub_df = DataFrames.groupby(new_df, ds_type[1])

        for i in new_sub_df
            if nrow(i) < ds_params.n_samples
                @warn "not enough samples, use replace=true"
                choosen_indexes = sample(rng, 1:nrow(i), ds_params.n_samples, replace=true)
            else
                choosen_indexes = sample(rng, 1:nrow(i), ds_params.n_samples, replace=false)
            end

            afe!(X, ds_params, ds_type[1], i, choosen_indexes)
        end

        save_jld2(X, string(ds_params.result_path, ds_params.name, "_male.jld2"))

        # gender, age no split
    else
        # create dataframe for audio features storing
        X = DataFrame()

        ## randomly pick a sample, save reference to a csv_dest and store wavfile
        for i in sub_df

            if nrow(i) < ds_params.n_samples
                @warn "not enough samples, use replace=true"
                choosen_indexes = sample(rng, 1:nrow(i), ds_params.n_samples, replace=true)
            else
                choosen_indexes = sample(rng, 1:nrow(i), ds_params.n_samples, replace=false)
            end

            afe!(X, ds_params, ds_type, i, choosen_indexes)
        end

        # if !debug
            save_jld2(X, string(ds_params.result_path, ds_params.name, ".jld2"))
        # else
        #     return X
        # end
    end
end
