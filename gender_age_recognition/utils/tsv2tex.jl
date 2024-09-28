using CSV, DataFrames, StatsBase, DataStructures

include("/home/paso/results/docs/tsv2tex.jl")

#--------------------------------------------------------------------------------------#
#                                         regex                                        #
#--------------------------------------------------------------------------------------#
model_header_regex = r"^(D[TF]),(\(.*\))(.*)$"
model_regex_tree = r"^DT,(\(.*\))(.*)$"
model_regex_forest = r"^DF,(\(.*\))(.*)$"

abs2rel_regex = r"[^/]+$"
dest_regex = r"([^_]*)_([^_]*)_([^_]*)_([^_]*)_?(.*)"

#--------------------------------------------------------------------------------------#
#                                         utils                                        #
#--------------------------------------------------------------------------------------#
function _is_model_column(colname::AbstractString, model::Regex; heads_separator::AbstractString = "@@@")
	return count(x -> !isnothing(x), match.(model, split(colname, heads_separator))) > 0
end
_is_model_column(colname::Symbol, model::Regex) = _is_model_column(string(colname), model::Regex)

_get_metric_cols(df::AbstractDataFrame, model::Regex) = Symbol.(filter(x -> _is_model_column(x, model), names(df)))

remove_last_col!(df::AbstractDataFrame) = select!(df, Not(ncol(df)))

_metric(colname::AbstractString) = _extract_info_from_model_column(colname)[3]
_metric(colname::Symbol) = Symbol(_metric(string(colname)))

#--------------------------------------------------------------------------------------#
#                                        filters                                       #
#--------------------------------------------------------------------------------------#
function _extract_info_from_model_column(
	colname::AbstractString;
	heads_separator::AbstractString = "@@@",
)::NTuple{3, String}
	if contains(colname, heads_separator)
		sp = split(colname, heads_separator)
		colname = sp[findfirst(_is_model_column, sp)]
	end

	m = match(model_header_regex, colname)

	isnothing(m) && throw(ErrorException("model not recognized in string $(colname)"))
	length(m.captures) < 2 && throw(ErrorException("params not recognized in string $(colname)"))
	length(m.captures) < 3 && throw(ErrorException("metric not recognized in string $(colname)"))

	return (m.captures...,)
end
_extract_info_from_model_column(colname::Symbol) = _extract_info_from_model_column(string(colname))

_metric(colname::AbstractString) = _extract_info_from_model_column(colname)[3]
_metric(colname::Symbol) = Symbol(_metric(string(colname)))

function keep_only_metrics!(df::AbstractDataFrame, metrics::AbstractVector{Symbol}, model::Regex)
	metriccols = _get_metric_cols(df, model)

	# to_keep = filter(x -> !(x in metriccols), Symbol.(names(df)))
	to_keep = Symbol[]
	for m in metrics
		append!(to_keep, filter(x -> Symbol(_metric(x)) == m, Symbol.(metriccols)))
	end

	return select!(df, to_keep)
end

function keep_only_metrics!(df::AbstractDataFrame, metrics::AbstractVector{<:AbstractString}, model::Regex)
	return keep_only_metrics!(df, Symbol.(metrics), model)
end
function keep_only_metrics!(df::AbstractDataFrame, metrics::AbstractString, model::Regex)
	return keep_only_metrics!(df, [metrics], model)
end
function keep_only_metrics!(df::AbstractDataFrame, metrics::Symbol, model::Regex)
	return keep_only_metrics!(df, [metrics], model)
end
function keep_only_metrics(df::AbstractDataFrame, args...; kwargs...)
	return keep_only_metrics!(deepcopy(df), args...; kwargs...)
end

#--------------------------------------------------------------------------------------#
#                                         main                                         #
#--------------------------------------------------------------------------------------#
# outdir = joinpath("/home/paso/Documents/Aclai/Datasets/Common_voice_ds/results/age2bins/spcds_age2bins_audioflux_full")
# inputfile = joinpath(outdir, "full_columns.tsv")
tablefile = "table.tex"
table_data = Tuple{Float64, Float64, Float64, Float64, Tuple{SubString{String}, SubString{String}, SubString{String}}, SubString{String}}[]

outdir = (joinpath("/home/paso/Documents/Aclai/Datasets/Common_voice_ds"))

# metrics = [:accuracy, :safe_macro_sensitivity, :safe_macro_specificity, :safe_macro_k]
metrics = [:K]

walkpath = "/home/paso/Documents/Aclai/Datasets/Common_voice_ds/results"
cd(walkpath)

for (root, _, files) in walkdir(".")
	for file in files
		if file == "full_columns.tsv"
			tsv = CSV.read(joinpath(root, file), DataFrame)
			# remove_last_col!(tsv)
            println(joinpath(root, file))

			# tenere i parametri dell'albero, la colonna migliore, e le parametrizzazioni che hanno la stessa media, magari in un dataframe a parte

			df_t = keep_only_metrics(tsv, metrics, model_regex_tree)
			df_f = keep_only_metrics(tsv, metrics, model_regex_forest)

			# find mean and std
			m_t = mean.(eachcol(df_t))
			m_f = mean.(eachcol(df_f))

			i_t = sortperm(m_t, rev = true)
			i_f = sortperm(m_f, rev = true)

			mean_t = m_t[1]
			mean_f = m_f[1]
			std_t = std(df_t[!, i_t[1]])
			std_f = std(df_f[!, i_f[1]])

			foldname = match(abs2rel_regex, root)
			dest = match(dest_regex, foldname.match)
			row_location = (dest[4], dest[2], dest[5])
			col_location = dest[3]

			push!(table_data, (mean_t, std_t, mean_f, std_f, row_location, col_location))
		end
	end
end

# col_pos = Dict("audio911" => 1, "audioflux" => 2, "matlab" => 3, "librosa911" => 4, "librosaflux" => 5, "mfcc911" => 6, "mfccflux" => 7)
# row_pos = Dict("gender" => 1, "age2bins" => 2, "age4bins" => 3, "age2split" => 4, "age4split" => 6)

col_pos = Dict("audio911" => 1, "audioflux" => 2, "matlab" => 3, "librosa911" => 4, "librosaflux" => 5)
row_pos = Dict("gender" => 1, "age2split" => 2)

split_offset = Dict("female" =>0, "male" => 1)
# row_bias = Dict("full" => 0, "optimized" => 1)

# index_df = [:audio911,:audioflux,:matlab, :librosa911, :librosaflux,:mfcc911, :mfccflux]
index_df = [:audio911,:audioflux,:matlab, :librosa911, :librosaflux]
d=(0.,0., 0., 0.)
# table_array = fill(d, 7, 7)
table_array = fill(d, 3, 5)
	
for i in table_data
	table_array[row_pos[i[5][2]]+(isempty(i[5][3]) ? 0 : split_offset[i[5][3]]), col_pos[i[6]]] = round.((i[1], i[2], i[3], i[4]), digits=2)
end

table_array = DataFrame(table_array, index_df)
# table_array[!, colname] 

latextable = latexdf(
    table_array,
    Symbol[];
    multirow_rotation = Int[],
    hide_param_names = true
)
tsv2tex_print(latextable, joinpath(outdir, tablefile))

## COMPILE
compile(joinpath(outdir, "main.tex"); show = true)


