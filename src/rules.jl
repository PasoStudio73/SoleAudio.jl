# using CSV, DataFrames, JLD2
# using Audio911
# using ModalDecisionTrees
# using SoleDecisionTreeInterface
# using MLJ, Random
# using StatsBase, Catch22
# using CategoricalArrays
# using Plots

# ---------------------------------------------------------------------------- #
#                                   regex                                      #
# ---------------------------------------------------------------------------- #
r_p_divide = r"(.*?)(  ↣  )(.*)"
r_cons = r"^\e\[\d+m▣\e\[0m\s*(.*?)\s*\n$"
r_p_ant = [
    r"\t",
    r"\e\[(?:1m|0m)",
    r"^\e\[\d+m▣ ", 
]
r_m_ant = [r"^SyntaxBranch:\s*", r"\e\[(?:1m|0m)", r"^SoleLogics.SyntaxBranch: *"]
r_var = r"\[V(\d+)\]"

format_float(x) = replace(x, r"(\d+\.\d+)" => s -> @sprintf("%.3f", parse(Float64, s)))

# ---------------------------------------------------------------------------- #
#                         interesting rule dataframe                           #
# ---------------------------------------------------------------------------- #
function interesting_rules(
    prop_sole_dt::Union{DecisionTree, Nothing}=nothing,
    modal_sole_dt::Union{DecisionTree, Nothing}=nothing;
    features::Symbol,
    variable_names::AbstractVector{String},
)
    if !isnothing(prop_sole_dt)
        metaconditions = get(propositional_feature_dict, features) do
            error("Unknown set of features: $features.")
        end

        irules = listrules(
            prop_sole_dt,
            min_lift = 1.0,
            # min_lift = 2.0,
            min_ninstances = 0,
            min_coverage = 0.10,
            normalize = true,
        );
        map(r->(consequent(r), readmetrics(r)), irules)
        p_irules = sort(irules, by=readmetrics)

        isempty(p_irules) && throw(ArgumentError("No interesting rules found."))
        
        p_X = DataFrame(antecedent=String[], consequent=String[]; [name => Vector{Union{Float64, Int}}() for name in keys(readmetrics(p_irules[1]))]...)
#         p_X = DataFrame(
#     antecedent = String[],
#     consequent = String[],
#     [name => Vector{Union{Float64, Int}}() for name in keys(readmetrics(p_irules[1]))]...
# )


        for i in eachrow(p_irules)
            a_c = match(r_p_divide, string(i[1]))
            antecedent, consequent = a_c[1], a_c[3]
            antecedent = reduce((s, r) -> replace(s, r => ""), r_p_ant, init=antecedent)
            antecedent = format_float(antecedent)
            push!(p_X, (antecedent, consequent, readmetrics(i[1])...))
        end
    else
        p_X = nothing
    end

    if !isnothing(modal_sole_dt)
        irules = listrules(
            modal_sole_dt,
            min_lift = 1.0,
            # min_lift = 2.0,
            min_ninstances = 0,
            min_coverage = 0.10,
            normalize = true,
            variable_names_map=variable_names
        )
        map(r->(r, readmetrics(r)), irules)
        m_irules = sort(irules, by=readmetrics)

        m_X = DataFrame(antecedent=String[], consequent=String[]; [name => Vector{Union{Float64, Int}}() for name in keys(readmetrics(m_irules[1]))]...)
        for i in eachrow(m_irules)
            consequent = match(r_cons, string(i[1].consequent))[1]
            # antecedent = foldl((s, r) -> replace(s, r => ""), r_antecedent, init=string(i[1].antecedent))
            # antecedent = replace(antecedent, r_variable => s -> begin
            #     m = match(r_variable, s)
            #     number = parse(Int, m[1])
            #     "[$(match(r_split, variable_names[number])[2])]"
            # end)
            antecedent = begin
                cleaned_string = reduce((s, r) -> replace(s, r => ""), r_m_ant, init=string(i[1].antecedent))
                replace(cleaned_string, r_var => s -> begin
                    number = parse(Int, match(r_var, s)[1])
                    "[$(match(r_split, variable_names[number])[2])]"
                end)
            end
            antecedent = format_float(antecedent)
            push!(m_X, (antecedent, consequent, readmetrics(i[1])...))
        end
    else
        m_X = nothing
    end

    vcat(p_X, m_X)
end