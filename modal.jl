# using CSV, DataFrames, JLD2
# using Audio911
# using ModalDecisionTrees
# using SoleDecisionTreeInterface
# using MLJ, Random
# using StatsBase, Catch22
# using CategoricalArrays
# using Plots

# ---------------------------------------------------------------------------- #
#                      handling of nan values functions                        #
# ---------------------------------------------------------------------------- #
function mean_longstretch1(x) Catch22.SB_BinaryStats_mean_longstretch1((x)) end
function diff_longstretch0(x) Catch22.SB_BinaryStats_diff_longstretch0((x)) end
function quantile_hh(x) Catch22.SB_MotifThree_quantile_hh((x)) end
function sumdiagcov(x) Catch22.SB_TransitionMatrix_3ac_sumdiagcov((x)) end

function histogramMode_5(x) Catch22.DN_HistogramMode_5((x)) end
function f1ecac(x) Catch22.CO_f1ecac((x)) end
function histogram_even_2_5(x) Catch22.CO_HistogramAMI_even_2_5((x)) end

function get_patched_feature(f::Base.Callable, polarity::Symbol)
    if f in [minimum, maximum, StatsBase.mean, median]
        f
    else
        @eval $(Symbol(string(f)*string(polarity)))
    end
end

nan_guard = [:std, :mean_longstretch1, :diff_longstretch0, :quantile_hh, :sumdiagcov, :histogramMode_5, :f1ecac, :histogram_even_2_5]

for f_name in nan_guard
    @eval (function $(Symbol(string(f_name)*"+"))(channel)
        val = $(f_name)(channel)

        if isnan(val)
            SoleData.aggregator_bottom(SoleData.existential_aggregator(≥), eltype(channel))
        else
            eltype(channel)(val)
        end
    end)
    @eval (function $(Symbol(string(f_name)*"-"))(channel)
        val = $(f_name)(channel)

        if isnan(val)
            SoleData.aggregator_bottom(SoleData.existential_aggregator(≤), eltype(channel))
        else
            eltype(channel)(val)
        end
    end)
end

const feature_dict = Dict(
    :catch9 => [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
        (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
        (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
        (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
        (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
        (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
        (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
        (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
    ],
    :minmax => [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
    ],
    :custom => [
        (≥, get_patched_feature(maximum, :+)),            (≤, get_patched_feature(maximum, :-)),
        # (≥, get_patched_feature(minimum, :+)),            (≤, get_patched_feature(minimum, :-)),
        # (≥, get_patched_feature(StatsBase.mean, :+)),     (≤, get_patched_feature(StatsBase.mean, :-)),
        # (≥, get_patched_feature(median, :+)),             (≤, get_patched_feature(median, :-)),
        (≥, get_patched_feature(std, :+)),                (≤, get_patched_feature(std, :-)),
        # (≥, get_patched_feature(mean_longstretch1, :+)),  (≤, get_patched_feature(mean_longstretch1, :-)),
        # (≥, get_patched_feature(diff_longstretch0, :+)),  (≤, get_patched_feature(diff_longstretch0, :-)),
        # (≥, get_patched_feature(quantile_hh, :+)),        (≤, get_patched_feature(quantile_hh, :-)),
        # (≥, get_patched_feature(sumdiagcov, :+)),         (≤, get_patched_feature(sumdiagcov, :-)),
        (≥, get_patched_feature(histogramMode_5, :+)),    (≤, get_patched_feature(histogramMode_5, :-)),
        (≥, get_patched_feature(f1ecac, :+)),             (≤, get_patched_feature(f1ecac, :-)),
        (≥, get_patched_feature(histogram_even_2_5, :+)), (≤, get_patched_feature(histogram_even_2_5, :-)),
    ]
)

# ---------------------------------------------------------------------------- #
#                                 modal analysis                               #
# ---------------------------------------------------------------------------- #
function modal_analisys(
    X::DataFrame,
    y::CategoricalArray;
    variable_names::AbstractVector{String},
    features::Symbol=:catch9,
    nwindows::Int=20,
    relative_overlap::Float64=0.05,
    train_ratio::Float64=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    # X_windowed = DataFrame([name => AbstractVector{<:AbstractFloat}[] for name in [match(r_select, v)[1] for v in variable_names]])
    X_windowed = DataFrame([name => Vector{Float64}[] for name in [match(r_select, v)[1] for v in variable_names]])
    for i in 1:nrow(X)
        audiowindowed = movingwindowmean.(Array(X[i, :]); nwindows = nwindows, relative_overlap = relative_overlap)
        push!(X_windowed, audiowindowed)
    end

    X_train, y_train, X_test, y_test = partitioning(X_windowed, y; train_ratio=train_ratio, rng=rng)

    @info("Modal analysis: train model...")
    metaconditions = get(feature_dict, features) do
        error("Unknown set of features: $features.")
    end

    learned_dt_tree = begin
        model = ModalDecisionTree(; relations = :IA7, features = metaconditions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        mach = machine(model, X_train, y_train) |> fit!
    end
    
    report(learned_dt_tree).printmodel(variable_names_map=variable_names)
    _, mtree = report(mach).sprinkle(X_test, y_test)
    # sole_dt = ModalDecisionTrees.translate(mtree)
    # printmodel(sole_dt; show_metrics = true, variable_names_map=variable_names)

    ModalDecisionTrees.translate(mtree)
end