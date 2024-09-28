using Pkg
Pkg.activate(".")
using Revise
using ModalDecisionTrees
using SimpleCaching
import Random

using Logging
using ResumableFunctions
using IterTools

using BenchmarkTools
using Statistics
using Test
# using Profile
# using PProf

using Catch22
using DataStructures

using DataFrames

using SHA
using Serialization
using FileIO
import JLD2
import Dates

using SoleBase: spawn, LogOverview

using SoleBase
# using SoleBase: utils
using MultiData
using SoleData
using SoleLogics
using SoleModels
using SoleData.DimensionalDatasets
using SoleData: TestOperator, MixedCondition
using ModalDecisionTrees
using ModalDecisionTrees: build_tree, build_forest
using ModalDecisionTrees: sprinkle

using Interpolations

include("lib.jl")
include("utils/metrics.jl")

include("scanner-utils/load-model.jl")
include("scanner-utils/feature-utils.jl")

include("functional-models/functional-models.jl")

id_f = identity
half_f(x) = ceil(Int, x / 2)
sqrt_f(x) = ceil(Int, sqrt(x))

function display_cm_as_row(cm::Main.ConfusionMatrix{Int64})
    "$(round(overall_accuracy(cm)*100,    digits=2))%\t" *
    "$(join(round.(cm.sensitivities.*100, digits=2), "%\t"))%\t" *
    "$(join(round.(cm.PPVs.*100,          digits=2), "%\t"))%\t" *
    "||\t" *
    # "$(round(cm.mean_accuracy*100, digits=2))%\t" *
    "$(round(cm.kappa*100, digits=2))%\t" *
    # "$(round(ModalDecisionTrees.macro_F1(cm)*100, digits=2))%\t" *
    # "$(round.(cm.accuracies.*100, digits=2))%\t" *
    "$(round.(cm.F1s.*100, digits=2))%\t" *
    "|||\t" *
    "$(round(safe_macro_sensitivity(cm)*100,  digits=2))%\t" *
    "$(round(safe_macro_specificity(cm)*100,  digits=2))%\t" *
    "$(round(safe_macro_PPV(cm)*100,          digits=2))%\t" *
    "$(round(safe_macro_NPV(cm)*100,          digits=2))%\t" *
    "$(round(safe_macro_F1(cm)*100,           digits=2))%\t" *
    # "$(round.(cm.sensitivities.*100, digits=2))%\t" *
    # "$(round.(cm.specificities.*100, digits=2))%\t" *
    # "$(round.(cm.PPVs.*100, digits=2))%\t" *
    # "$(round.(cm.NPVs.*100, digits=2))%\t" *
    # "|||\t" *
    # "$(round(ModalDecisionTrees.macro_weighted_F1(cm)*100, digits=2))%\t" *
    # # "$(round(ModalDecisionTrees.macro_sensitivity(cm)*100, digits=2))%\t" *
    # "$(round(ModalDecisionTrees.macro_weighted_sensitivity(cm)*100, digits=2))%\t" *
    # # "$(round(ModalDecisionTrees.macro_specificity(cm)*100, digits=2))%\t" *
    # "$(round(ModalDecisionTrees.macro_weighted_specificity(cm)*100, digits=2))%\t" *
    # # "$(round(ModalDecisionTrees.macro_PPV(cm)*100, digits=2))%\t" *
    # "$(round(ModalDecisionTrees.macro_weighted_PPV(cm)*100, digits=2))%\t" *
    # # "$(round(ModalDecisionTrees.macro_NPV(cm)*100, digits=2))%\t" *
    # "$(round(ModalDecisionTrees.macro_weighted_NPV(cm)*100, digits=2))%\t" *
    ""
end

include("caching.jl")
include("scanner-utils/install.jl")
include("scanner-utils/table-printer.jl")
include("scanner-utils/progressive-iterator-manager.jl")

# UNIMODAL DATASETS
@safeconst GenericUniDataset = Union{
    AbstractArray{<:Number}, # Cube (deprecating...)
    SoleData.AbstractDimensionalDataset,
    AbstractDataFrame,
    SoleModels.AbstractLogiset,
}

# MULTIMODAL DATASETS
@safeconst GenericMultiDataset = Union{
    # TODO add multi-dimensional-datasets?
    MultiLogiset,
}

@safeconst GenericDataset = Union{
    GenericUniDataset,
    GenericMultiDataset,
}

include("datasets/dataset-export.jl")
include("datasets/dataset-utils.jl")
include("datasets/datasets.jl")
# include("datasets/dataset-analysis.jl")
include("datasets/apply-multimodal-modes.jl")

include("datasets/feature-selection.jl")


############################################################################################
############################################################################################
############################################################################################

relations_dict = Dict(
    "-" => AbstractRelation[],
    "RCC5" => [globalrel, SoleLogics.RCC5Relations...],
    "RCC8" => [globalrel, SoleLogics.RCC8Relations...],
    "IA" => [globalrel, SoleLogics.IARelations...],
    "IA7" => [globalrel, SoleLogics.IA7Relations...],
    "IA3" => [globalrel, SoleLogics.IA3Relations...],
    "IA2D" => [globalrel, SoleLogics.IA2DRelations...],
    #
    # "o_ALLiDxA" => ([SoleLogics.IA_AA, SoleLogics.IA_LA, SoleLogics.IA_LiA, SoleLogics.IA_DA]),
    #
    "gS" => [globalrel, SoleLogics.SuccessorRel],
    "gSG" => [globalrel, SoleLogics.SuccessorRel, SoleLogics.GreaterRel],
    #
    "S" => [SoleLogics.SuccessorRel],
    "SG" => [SoleLogics.SuccessorRel, SoleLogics.GreaterRel],
    #
    "gSPGL" => [globalrel, SoleLogics.SuccessorRel, SoleLogics.PredecessorRel, SoleLogics.GreaterRel, SoleLogics.LesserRel],
    "gSP" => [globalrel, SoleLogics.SuccessorRel, SoleLogics.PredecessorRel],
    "gGL" => [globalrel, SoleLogics.GreaterRel, SoleLogics.LesserRel],
    #
    "gMSPGL" => [globalrel, SoleLogics.MinRel, SoleLogics.MaxRel, SoleLogics.SuccessorRel, SoleLogics.PredecessorRel, SoleLogics.GreaterRel, SoleLogics.LesserRel],
)

############################################################################################
############################################################################################
############################################################################################

get_is_regression_problem((X, Y)::Tuple) = (eltype(Y) <: AbstractFloat)
get_is_regression_problem(((X_train, Y_train), (X_test, Y_test))::Tuple{Tuple,Tuple}) = get_is_regression_problem((X_train, Y_train))

# function get_nmodalities(X::AbstractVector{<:AbstractDimensionalDataset})
function get_nmodalities(X::AbstractVector{<:AbstractVector})
    length(X)
end
function get_nmodalities(X::AbstractDimensionalDataset)
    1
end
get_nmodalities((X, Y)::Tuple) = get_nmodalities(X)
get_nmodalities(((X_train, Y_train), (X_test, Y_test))::Tuple{Tuple,Tuple}) = get_nmodalities((X_train, Y_train))


function fix_multimodaldataset(X::AbstractVector{<:AbstractDimensionalDataset}, issurelymultimodal=false)

    @warn "Multimodal dataset encountered"
    X
end
function fix_multimodaldataset(X::AbstractDimensionalDataset, issurelymultimodal=false)

    if length(X) < 10 && !issurelymultimodal
        # Assuming multimodal
        @warn "A vector of cubes with small length was provided ($(typeof(X)) with length $(length(X)))." *
              " This is probably intended to be the number of modalities, rather than the number of instances." *
              " However cube datasets are deprecating:" *
              " please, unroll each cube via `eachslice(X; dims=ndims(X))`."
        " Otherwise, please specify `issurelymultimodal`."
        map(mod -> eachslice(mod; dims=ndims(mod)), X)
    else
        @warn "Unimodal dataset encountered"
        [X]
    end
end

function fix_multimodaldataset(X::AbstractArray, issurelymultimodal=false)

    @warn "Cubes are deprecating, but a $(typeof(X)) dataset was provided. Please provide `eachslice(X; dims=ndims(X))` instead."
    
    @info "check"
    println(ndims(X))
    X
    # [eachslice(X; dims=ndims(X))]
    # [collect(slice) for slice in eachslice(X; dims=ndims(X))]
    # [vec(slice) for slice in eachslice(X; dims=3)]
end

fix_multimodaldataset((X, Y)::Tuple, args...) = (fix_multimodaldataset(X, args...), Y)
fix_multimodaldataset(((X_train, Y_train), (X_test, Y_test))::Tuple{Tuple,Tuple}, args...) = fix_multimodaldataset((X_train, Y_train), args...), fix_multimodaldataset((X_test, Y_test), args...)

# Slice & split the dataset according to dataset_slices & split_threshold
# The instances for which the full SupportedScalarLogiset is computed are either all, or the ones specified for training;
# This depends on whether the dataset is already splitted or not.
# Dataset is to be splitted
@resumable function generate_splits(
    dataset::Tuple, # (X,y) -> ((X,y),(X,y))
    split_threshold,
    round_dataset_to_datatype,
    save_datasets,
    run_name,
    dataset_slices,
    data_savedir,
    makelogisets_fun,
)
    X, Y = dataset

    # Apply scaling
    if round_dataset_to_datatype != false
        X, Y = round_dataset((X, Y), round_dataset_to_datatype)
    end
    # Compute mffmd for the full set of instances
    X_to_train, X_to_test = makelogisets_fun(X, X)

    # TODO
    # if save_datasets
    #   train = (X_to_train, Y)
    #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-train.jld" train
    #   test = (X_to_test, Y)
    #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-test.jld" test
    # end

    for (slice_id, dataset_slice) in dataset_slices

        print("Dataset_slice = ")
        if isnothing(dataset_slice)
            println("nothing")
        elseif isa(dataset_slice, AbstractDatasetSlice)
            println("($(length(dataset_slice))) -> $(dataset_slice)")
        elseif isa(dataset_slice, AbstractDatasetSplit)
            println("($(length.(dataset_slice))) -> $(dataset_slice)")
        else
            throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
        end

        # Slice instances (for balancing, for example)
        X_train_slice, Y_train_slice, X_test_slice, Y_test_slice, dataset_split = begin
            if isnothing(dataset_slice) || isa(dataset_slice, AbstractDatasetSlice)
                @assert split_threshold !== false "`split_threshold` needed to split stratified dataset"
                # Use two different structures for train and test
                X_to_train_slice, X_to_test_slice, Y_slice, dataset_slice =
                    if isnothing(dataset_slice)
                        (
                            X_to_train,
                            X_to_test,
                            Y,
                            collect(1:length(Y))
                        )
                    else
                        (
                            slicedataset(X_to_train, dataset_slice; return_view=true),
                            slicedataset(X_to_test, dataset_slice; return_view=true),
                            Y[dataset_slice],
                            dataset_slice
                        )
                    end

                # Split in train/test
                (X_train_slice, Y_train_slice), _ = traintestsplit((X_to_train_slice, Y_slice), split_threshold; return_view=true)
                _, (X_test_slice, Y_test_slice) = traintestsplit((X_to_test_slice, Y_slice), split_threshold; return_view=true)

                dataset_split = traintestsplit(dataset_slice, split_threshold)

                (X_train_slice, Y_train_slice, X_test_slice, Y_test_slice, dataset_split)
            elseif isa(dataset_slice, AbstractDatasetSplit)
                dataset_split = dataset_slice
                train_idxs, test_idxs = dataset_split
                println("train slice: $(dataset_slice[1])")
                println("test  slice: $(dataset_slice[2])")

                @assert slice_id == 0 || length(intersect(Set(train_idxs), Set(test_idxs))) == 0 "Non-empty intersection between train and test slice: $(intersect(Set(train_idxs), Set(test_idxs)))"

                @assert isone(split_threshold) || iszero(split_threshold) || (isa(split_threshold, Bool) && !split_threshold) "Cannot set a split_threshold (value: $(split_threshold)) when specifying a split dataset_slice (type: $(typeof(dataset_slice)))"
                (
                    slicedataset(X_to_train, train_idxs; return_view=true),
                    Y[train_idxs],
                    slicedataset(X_to_test, test_idxs; return_view=true),
                    Y[test_idxs],
                    dataset_split,
                )
            else
                throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
            end
        end

        # if save_datasets
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-dataset_slice.jld" dataset_slice
        #   sliced = (X_to_train_slice, Y_slice)
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced.jld" sliced
        #   sliced_train = (X_train_slice, Y_train_slice)
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_train.jld" sliced_train
        #   sliced_test = (X_test_slice, Y_test_slice)
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_test.jld" sliced_test
        # end

        @yield ((X_to_train, X), slice_id, ((X_train_slice, Y_train_slice), (X_test_slice, Y_test_slice)), dataset_split)
    end
end

# Dataset is already splitted
@resumable function generate_splits(
    dataset::Tuple{<:Tuple,<:Tuple}, # ((X_train,y_train),(X_test,y_test))
    split_threshold,
    round_dataset_to_datatype,
    save_datasets,
    run_name,
    dataset_slices,
    data_savedir,
    makelogisets_fun,
)

    ((X_train, Y_train), (X_test, Y_test)) = dataset

    # Apply scaling
    if round_dataset_to_datatype != false
        (X_train, Y_train), (X_test, Y_test) = round_dataset(((X_train, Y_train), (X_test, Y_test)), round_dataset_to_datatype)
    end

    # Compute mffmd for the training instances
    X_to_train, X_to_test = makelogisets_fun(X_train, X_test)

    # if save_datasets
    #   train = (X_to_train, Y_train)
    #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-train.jld" train
    #   test = (X_to_test, Y_test)
    #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-test.jld" test
    # end

    # Slice *training* instances (for balancing, for example)
    for (slice_id, dataset_slice) in dataset_slices

        print("Dataset_slice = ")
        if isnothing(dataset_slice)
            println("nothing")
        elseif isa(dataset_slice, AbstractDatasetSlice)
            println("($(length(dataset_slice))) -> $(dataset_slice)")
        elseif isa(dataset_slice, AbstractDatasetSplit)
            println("($(length.(dataset_slice))) -> $(dataset_slice)")
        else
            throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
        end

        X_train_slice, Y_train_slice, X_test_slice, Y_test_slice, dataset_split =
            if isnothing(dataset_slice)
                dataset_split = collect(1:length(Y_train)), collect(1:length(Y_test))
                (X_to_train, Y_train, X_to_test, Y_test, dataset_split)
            elseif isa(dataset_slice, AbstractDatasetSlice)
                dataset_split = (dataset_slice, 1:length(Y_test))
                @assert split_threshold !== false "`split_threshold` needed to split stratified dataset"
                (
                    slicedataset(X_to_train, dataset_slice; return_view=true),
                    Y_train[dataset_slice],
                    X_to_test,
                    Y_test,
                    dataset_split,
                )
            elseif isa(dataset_slice, AbstractDatasetSplit)
                dataset_split = dataset_slice
                train_idxs, test_idxs = dataset_split
                @assert isone(split_threshold) || iszero(split_threshold) || (isa(split_threshold, Bool) && !split_threshold) "Cannot set a split_threshold (value: $(split_threshold)) when specifying a split dataset_slice (type: $(typeof(dataset_slice)))"
                throw_n_log("TODO expand code. When isa(dataset_slice, AbstractDatasetSplit) and the dataset is already splitted, must also test on the validation data! Maybe when the dataset is already splitted into ((X_train, Y_train), (X_test, Y_test)), join it and create a dummy dataset_slice")
                (
                    slicedataset(X_to_train, train_idxs; return_view=true),
                    Y_train[train_idxs],
                    slicedataset(X_to_train, test_idxs; return_view=true),
                    Y_train[test_idxs],
                    dataset_split,
                )
            else
                throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
            end

        # if save_datasets
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-dataset_slice.jld" dataset_slice
        #   sliced_train = (X_train_slice, Y_train_slice)
        #   JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_train.jld" sliced_train
        # end

        @yield ((X_to_train, X_train), slice_id, ((X_train_slice, Y_train_slice), (X_test_slice, Y_test_slice)), dataset_split)
    end
end

# TODO fix and remove
function makelogiset(
    dataset_type_str,
    data_modal_args::AbstractVector{<:NamedTuple},
    dataset,
    modal_args,
    save_datasets,
    timing_mode,
    data_savedir,
    form
)
    println("This method for `makelogiset` is deprecated!")
    return makelogiset(
        dataset,
        data_modal_args;
        form=form,
        dataset_type_str=dataset_type_str,
        save_datasets=save_datasets,
        timing_mode=timing_mode,
        data_savedir=data_savedir,
    )
end

function makelogiset(
    dataset,
    data_modal_args::AbstractVector{<:NamedTuple};
    form=:supportedlogiset,
    dataset_type_str="—",
    save_datasets=false,
    timing_mode=:time,
    data_savedir=".",
)
    println("Computing $(dataset_type_str) dataset (form = $(form), save_datasets = $(save_datasets), timing_mode = $(timing_mode))...")

    # if ! (mixed_conditions isa AbstractVector{AbstractVector})
    mixed_conditions = [data_modal_args[i_modality].mixed_conditions for i_modality in 1:length(data_modal_args)]
    # end
    # if ! (relations isa AbstractVector{AbstractVector})
    relations = [data_modal_args[i_modality].relations for i_modality in 1:length(data_modal_args)]
    # end

    # compute_globmemoset =
    # WorldType != OneWorld && (
    #     (modal_args.allow_global_splits || (modal_args.initconditions == ModalDecisionTrees.start_without_world))
    #         || ((modal_args.initconditions isa AbstractVector) && modal_args.initconditions[i_modality] == ModalDecisionTrees.start_without_world)
    # ) -> map(_relations->[globalrel, _relations...], relations)

    @show mixed_conditions
    @show relations

    function _makelogiset(X, mixed_conditions, relations)
        kwargs = (; print_progress=true)

        f = begin
            if form == :passive
                (X) -> (X)
            elseif form == :logiset
                (X) -> scalarlogiset(X, mixed_conditions; kwargs...)
            elseif form == :supportedlogiset
                (X) -> scalarlogiset(
                    X,
                    mixed_conditions;
                    use_onestep_memoization=true,
                    onestep_precompute_relmemoset=true,
                    relations=relations, kwargs...
                )
            elseif form == :supportedlogiset_with_memoization
                (X) -> scalarlogiset(
                    X,
                    mixed_conditions;
                    use_onestep_memoization=true,
                    onestep_precompute_relmemoset=false,
                    relations=relations, kwargs...
                )
            else
                throw_n_log("Unexpected value for form: $(form)!")
            end
        end

        if timing_mode == :none
            f(X)
        elseif timing_mode == :time
            @time f(X)
        elseif timing_mode == :btime
            @btime f($X)
        end
    end

    if save_datasets
        @cachefast "$(dataset_type_str)_logiset" data_savedir (dataset, mixed_conditions, relations) _makelogiset
    else
        _makelogiset(dataset, mixed_conditions, relations)
    end
end

# This function transforms bare AbstractDimensionalDataset's into modal datasets in the form of implicit or featmodal dataset
#  The train dataset, unless use_training_form, is transformed in featmodal form, which is optimized for training.
#  The test dataset is kept in implicit form
function makelogisets(X_train, X_test, data_modal_args, modal_args, use_training_form, data_savedir, timing_mode, save_datasets, use_test_form)

    # @show typeof(X_train)
    # @show typeof(X_test)
    # @show typeof(X_train[1])
    # @show typeof(X_train[1][1])
    # @show typeof(eachmodality(X_train))
    # @show (nmodalities(X_train))
    # @show data_modal_args

    @assert !dataset_has_nonevalues(X_train) "dataset_has_nonevalues(X_train)"
    @assert !dataset_has_nonevalues(X_test) "dataset_has_nonevalues(X_test)"

    # Split-level hybrid: use train split for training autoencoders
    for (i_modality, _X_train) in enumerate(eachmodality(X_train))
        @show i_modality
        @show data_modal_args[i_modality].mixed_conditions
        data_modal_args[i_modality] = merge(data_modal_args[i_modality], (;
            mixed_conditions=Vector{MixedCondition}(collect(Iterators.flatten([
                begin
                    if mf isa Tuple && mf[1] == :pretrained_autoencoder
                        autoenc_data_params = mf[2]
                        autoenc_params = mf[3]
                        p = shuffle(autoenc_data_params.rng, 1:ninstances(_X_train))
                        __X_train = slicedataset(_X_train, p)
                        ___X_train, ___X_valid = traintestsplit(__X_train, autoenc_data_params.split_threshold)

                        autoencs = [
                            begin
                                println("Training autoencoder for variable $(i_var)/$(nvariables(___X_train))")
                                FunctionalModels.train_model(
                                    scalarlogiset([___X_train[:, [i_var], :]]),
                                    scalarlogiset([___X_valid[:, [i_var], :]]),
                                    autoenc_data_params.rng;
                                    (autoenc_params)...
                                )
                            end for i_var in 1:nvariables(___X_train)
                        ]

                        function autoenc_component(autoenc, i)
                            return (x) -> autoenc(x)[i]
                        end
                        [UnivariateFeature{FunctionalModels.NN_ELTYPE}(i_var, autoenc_component(autoenc, i))
                         for i in 1:(autoenc_params.code_size)
                         for (i_var, autoenc) in enumerate(autoencs)]
                    else
                        [mf]
                    end
                end for mf in data_modal_args[i_modality].mixed_conditions
            ])))
        )
        )
    end

    # The test dataset is kept in its implicit form
    X_test = makelogiset("test", data_modal_args, X_test, modal_args, save_datasets, timing_mode, data_savedir, use_test_form)

    # The train dataset is either kept in implicit form, or processed into stump form (which allows for optimized learning)
    X_train =
        if use_training_form in [:passive, :logiset, :supportedlogiset, :supportedlogiset_with_memoization]
            if save_datasets
                if use_training_form == :supportedlogiset_with_memoization
                    @cachefast_skipsave "training_dataset" data_savedir ("train", data_modal_args, X_train, modal_args, false, timing_mode, data_savedir, use_training_form) makelogiset
                else
                    @cachefast "training_dataset" data_savedir ("train", data_modal_args, X_train, modal_args, false, timing_mode, data_savedir, use_training_form) makelogiset
                end
            else
                makelogiset(("train", data_modal_args, X_train, modal_args, save_datasets, timing_mode, data_savedir, use_training_form)...)
            end
        else
            throw_n_log("Unexpected value for use_training_form: $(use_training_form)!")
        end

    X_train, X_test
end


function exec_scan(
    params_namedtuple::NamedTuple,
    dataset::Tuple;
    ### Training params
    train_seed=1,
    modal_args=(),
    tree_args=[],
    nsdt_args=[],
    n_nsdt_folds=nothing,
    nsdt_training_args::Union{NamedTuple,AbstractVector}=[(;)],
    nsdt_finetuning_args::Union{NamedTuple,AbstractVector}=nsdt_training_args,
    forest_args=[],
    n_forest_runs=1,
    optimize_forest_computation=false,
    ### Dataset params
    split_threshold::Union{Bool,AbstractFloat}=1.0,
    data_modal_args::Union{NamedTuple,AbstractVector{<:NamedTuple}}=NamedTuple(),
    dataset_slices::Union{
        AbstractVector{<:Tuple{<:Any,<:AbstractDatasetSplitOrSlice}},
        AbstractVector{<:AbstractDatasetSplitOrSlice},
        Nothing}=nothing,
    round_dataset_to_datatype::Union{Bool,Type}=false,
    use_training_form=:supportedlogiset_with_memoization,
    dataset_filepath::String="",
    ### Run params
    results_dir::String,
    data_savedir::Union{String,Nothing}=".",
    model_savedir::Union{String,Nothing}=".",
    legacy_gammas_check=false,
    log_level=nothing, # TODO remove this obsolete parameter
    logger=ConsoleLogger(stderr, SoleBase.LogOverview), # TODO remove this as well?
    timing_mode::Symbol=:time,
    ### Misc
    # best_rule_params                = [(t=.8, min_confidence=0.6, min_support=0.1), (t=.65, min_confidence=0.6, min_support=0.1)],
    issurelymultimodal::Bool=false,
    save_datasets::Bool=false,
    skip_training::Bool=false,
    resilient_mode::Bool=false,
    callback::Function=identity,
    other_columns::Union{AbstractVector,Nothing}=nothing,
    use_test_form::Symbol=:passive,
)

    @assert timing_mode in [:none, :profile, :time, :btime] "Unknown timing_mode!"
    @assert !legacy_gammas_check "legacy_gammas_check parameter is deprecated!" # TODO remove

    run_name = join([replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)], ",")


    ##############################################################################
    ##############################################################################
    ##############################################################################

    println()
    println("executing run '$run_name'...")
    println("dataset type = ", typeof(dataset))

    ##############################################################################
    ##############################################################################
    ##############################################################################

    dataset = fix_multimodaldataset(dataset, issurelymultimodal)

    begin
        nmodalities = get_nmodalities(dataset)
        println("# modalities = $(nmodalities)")
        @assert nmodalities < 100 "Mh. Something's not right with the dataset, when nmodalities = $(nmodalities)"

        # Force data_modal_args to be an array of itself (one for each modality)
        if isa(data_modal_args, NamedTuple)
            data_modal_args = [deepcopy(data_modal_args) for i in 1:nmodalities]
        end

        @assert length(data_modal_args) == nmodalities "Mismatching number of `data_modal_args` provided ($(length(data_modal_args))). Must be either one, or the number of modalities ($(nmodalities)): $(data_modal_args)"
    end

    if !isa(nsdt_training_args, AbstractVector)
        nsdt_training_args = [nsdt_training_args]
    end
    nsdt_training_args = Vector{NamedTuple}(nsdt_training_args)

    if !isa(nsdt_finetuning_args, AbstractVector)
        nsdt_finetuning_args = [nsdt_finetuning_args]
    end
    nsdt_finetuning_args = Vector{NamedTuple}(nsdt_finetuning_args)

    class_names = collect(get_class_names(dataset))
    is_regression_problem = get_is_regression_problem(dataset)

    if !isnothing(log_level)
        println("Warning! scanner.log_level parameter is obsolete. Use logger = ConsoleLogger(stderr, $(log_level)) instead!")
        logger = ConsoleLogger(stderr, log_level)
    end

    ##############################################################################
    ##############################################################################
    # Output files
    ##############################################################################
    ##############################################################################

    full_output_filepath = results_dir * "/full_columns.tsv"

    results_col_sep = "\t"

    base_metrics_names = begin
        if is_regression_problem
            [
                "train_cor",
                "train_MAE",
                "train_RMSE",
                "cor",
                "MAE",
                "RMSE",
            ]
        else
            [
                "train_accuracy",
                "K",
                "accuracy",
                "macro_sensitivity",
                "safe_macro_sensitivity",
                "safe_macro_specificity",
                "safe_macro_PPV",
                "safe_macro_NPV",
                "safe_macro_F1",
            ]
        end
    end

    _tree_columns = [base_metrics_names..., "nnodes", "nleaves", "height", "modalheight",]
    _forest_columns = [base_metrics_names..., "oob_error", "ntrees", "nnodes",]

    tree_columns = ["hashpath", "t", _tree_columns...,]
    nsdt_columns = [
        "hashpath", "t",
        _tree_columns...,
        ["σ² $(n)" for n in _tree_columns]...,
    ]
    forest_columns = [
        "hashpath", "t",
        _forest_columns...,
        ["σ² $(n)" for n in _forest_columns]...,
    ]

    other_functions = OrderedDict(
        # Common
        [
            "datetime" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> string(now()),
            "# threads" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> Threads.nthreads(),
            "# insts (TRAIN)" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> length(Y_train),
            "# insts (TEST)" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> length(Y_test),
        ]...,
        # Classification
        (!is_regression_problem ? [
            "class_counts (TRAIN)" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> get_ungrouped_class_counts(Y_train),
            "class_counts (TEST)" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> get_ungrouped_class_counts(Y_test),
        ] : [])...,
        [
            "dataset_filepath" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> (dataset_filepath),
            "train idxs" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> dataset_slice[1],
            "test idxs" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> dataset_slice[2],
            #
            "nvariables" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> nvariables.(X_test),
            "nfeatures" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> (X_train isa SoleData.MultiLogiset ? nfeatures.(eachmodality(X_train)) : (X_train isa SoleModels.AbstractLogiset ? nfeatures(X_train) : "? (X_train of type $(typeof(X_train)))")),
            "maxchannelsize" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> (X_test isa Union{SoleData.DimensionalDatasets.UniformFullDimensionalLogiset,SoleData.AbstractDimensionalDataset} ? maxchannelsize(X_test) : (X_test isa AbstractVector ? maxchannelsize.(X_test) : "? (X_test of type $(typeof(X_test)))")),
            "size" => (X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath) -> (X_test isa AbstractVector ? Base.size.(X_test) : "? (X_test of type $(typeof(X_test)))"),
        ]...,
    )

    if isnothing(other_columns)
        other_columns = collect(keys(other_functions))
    else
        @assert all([col_name in keys(other_functions) for col_name in other_columns]) "Unknown column encountered in other_columns = $(other_columns)"
    end

    condition_columns = String["Dataseed", (params_namedtuple |> keys .|> string)..., other_columns...]

    model_columns = begin
        model_columns = String[]
        for this_tree_args in tree_args
            for this_column in tree_columns
                push!(model_columns, "DT,$(Tuple(this_tree_args))$(this_column)")
            end
        end

        for this_nsdt_training_args in nsdt_finetuning_args
            for this_nsdt_finetuning_args in nsdt_training_args
                for this_nsdt_args in nsdt_args
                    for this_column in nsdt_columns
                        push!(model_columns, "NSDT,$(Tuple(this_nsdt_args)),$(this_nsdt_training_args),$(this_nsdt_finetuning_args)$(this_column)")
                    end
                end
            end
        end

        for this_forest_args in forest_args
            for this_column in forest_columns
                push!(model_columns, "DF,$(Tuple(this_forest_args))$(this_column)")
            end
        end
        model_columns
    end

    columns = String[condition_columns..., model_columns...]

    # If the output files do not exists initilize them
    print_result_head(
        full_output_filepath,
        columns;
        separator=results_col_sep,
    )

    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

    function go_tree(slicemap, X_train, Y_train, X_test, Y_test, this_args, rng; post_pruning_args)
        slice_id, dataset_split = slicemap

        started = Dates.now()
        model =
            if timing_mode == :none
                build_tree(X_train, Y_train; this_args..., modal_args..., rng=rng)
                # elseif timing_mode == :profile
                # @profile build_tree(X_train, Y_train; this_args..., modal_args..., rng = rng)
            elseif timing_mode == :time
                @time build_tree(X_train, Y_train; this_args..., modal_args..., rng=rng)
            elseif timing_mode == :btime
                @btime build_tree($X_train, $Y_train; $this_args..., $modal_args..., rng=$rng)
            end
        time_elapsed = Dates.now() - started
        println("Train tree:")
        print(model)

        unique_post_pruning_args = (post_pruning_args isa NamedTuple)
        if unique_post_pruning_args
            post_pruning_args = [post_pruning_args]
        end

        ret = [
            begin
                if !(length(post_pruning_args) == 1 && this_pruning_args == (;))
                    println(" Post-pruning arguments:")
                    println(this_pruning_args)
                end
                started = Dates.now()
                model_pruned = prune(model; this_pruning_args...)
                model_prunedt = Dates.now() - started

                println(" Pruned tree $(i_post_pruning_args) / $(length(post_pruning_args)):")
                println(model_pruned)

                model_save_path = ""
                hash = get_hash_sha256(model_pruned)
                if !isnothing(model_savedir)
                    model_save_path = model_savedir * "/model_" * hash * ".jld"
                    mkpath(dirname(model_save_path))

                    checkpoint_stdout("Saving tree to file $(model_save_path)...")
                    JLD2.@save model_save_path model_pruned
                end

                m_train = compute_metrics(Y_train, apply_model(model_pruned, X_train))

                println(" # test instances = $(ninstances(X_test))")

                preds, model_pruned_test = sprinkle(model_pruned, X_test, Y_test)
                m = compute_metrics(Y_test, preds)

                # If not full-training
                if sort(dataset_split[1]) != sort(dataset_split[2])
                    println("Test tree:")
                    ModalDecisionTrees.printmodel(model_pruned_test)
                end

                m_str = begin
                    if is_regression_problem
                        println()
                        display(m)
                    else
                        println(m.cm)
                        display_cm_as_row(m.cm)
                    end
                end

                println("RESULT:\t$(run_name)\t$(slice_id)\t$(this_args)\t$(modal_args)\t$(this_pruning_args)\t|\t$(m_str)\t$(model_save_path)")

                (model_pruned_test, m_train, m, (i_post_pruning_args == 1 ? time_elapsed + model_prunedt : model_prunedt), hash)

                model_pruned_tests = DTree[model_pruned_test]
                m_trains = NamedTuple[m_train]
                ms = NamedTuple[m]
                ts = Dates.Millisecond[(i_post_pruning_args == 1 ? time_elapsed + model_prunedt : model_prunedt)]
                hashes = String[hash]

                # println("nodes: ($(nnodes(model_pruned)), height: $(height(model_pruned)))")
                (model_pruned_tests, m_trains, ms, ts, hashes)
            end for (i_post_pruning_args, this_pruning_args) in enumerate(post_pruning_args)]

        if unique_post_pruning_args
            ret[1]
        else
            ret
        end
    end

    function go_nsdt(slicemap, X_train, Y_train, X_test, Y_test, this_args, rng; post_pruning_args, this_nsdt_training_args, this_nsdt_finetuning_args)
        @show (this_args)
        @show Base.structdiff(this_args, (; hybrid_type=nothing))
        return begin
            if haskey(this_args, :hybrid_type)
                if this_args.hybrid_type == :leaflevel
                    _go_nsdt_leaflevel(slicemap, X_train, Y_train, X_test, Y_test, Base.structdiff(this_args, (; hybrid_type=nothing)), rng; post_pruning_args, this_nsdt_training_args, this_nsdt_finetuning_args)
                elseif this_args.hybrid_type == :rootlevel
                    _go_nsdt_rootlevel(slicemap, X_train, Y_train, X_test, Y_test, Base.structdiff(this_args, (; hybrid_type=nothing)), rng; post_pruning_args, this_nsdt_training_args, this_nsdt_finetuning_args)
                else
                    error("Unknown hybrid_type encountered in nsdt_args: $(this_args.hybrid_type)")
                end
            else
                error("Unspecified hybrid_type in nsdt_args.")
            end
        end
    end


    function _go_nsdt_rootlevel(slicemap, X_train, Y_train, X_test, Y_test, this_args, rng; post_pruning_args, this_nsdt_training_args, this_nsdt_finetuning_args)
        slice_id, dataset_split = slicemap
        rng1 = spawn(rng)
        rng2 = spawn(rng)
        rng3 = spawn(rng)
        rng4 = spawn(rng)

        @assert !isnothing(n_nsdt_folds) "Please specificy n_nsdt_folds!"

        # @assert length(this_nsdt_finetuning_args) == 1 && this_nsdt_finetuning_args[1] == (;) "$this_nsdt_finetuning_args"

        cv_dataset_slices = balanced_cv_dataset_slices((X_train, Y_train), n_nsdt_folds, rng1; strict=false)
        println("cv_dataset_slices: $(cv_dataset_slices)")

        # Train neural networks
        cv_base_functional_models = Dict{DatasetSplit,FunctionalModels.FunctionalModel}([
            begin
                training_idxs, validation_idxs = cv_dataset_slice
                (X_train_t, Y_train_t) = slicedataset((X_train, Y_train), training_idxs)
                (X_train_v, Y_train_v) = slicedataset((X_train, Y_train), validation_idxs)
                cv_base_functional_model = FunctionalModels.train_model((X_train_t, Y_train_t), (X_train_v, Y_train_v), class_names, rng2; this_nsdt_training_args...)
                cv_dataset_slice => cv_base_functional_model
                # (cv_dataset_slice => @cachefast "nn" model_savedir ((X_train_t, Y_train_t), (X_train_v, Y_train_v), class_names, rng2) this_nsdt_training_args FunctionalModels.train_model)
            end for cv_dataset_slice in cv_dataset_slices
        ])

        # Obtain instance weights
        Wss = [Flux.softmax(func_model(X_train)) for (cv_dataset_slice, func_model) in cv_base_functional_models]

        started = Dates.now()
        models = hcat([
            begin
                [
                    begin
                        if timing_mode == :none
                            build_tree(X_train, Y_train, W; this_args..., modal_args..., rng=rng3)
                            # elseif timing_mode == :profile
                            # @profile build_tree(X_train, Y_train, W; this_args..., modal_args..., rng = rng3)
                        elseif timing_mode == :time
                            @time build_tree(X_train, Y_train, W; this_args..., modal_args..., rng=rng3)
                        elseif timing_mode == :btime
                            @btime build_tree($X_train, $Y_train, $W; $this_args..., $modal_args..., rng=$rng3)
                        end
                    end for W in eachrow(Ws)
                ]
            end for Ws in Wss
        ]...)
        # @show size(models) # (code_size × n_folds)
        time_elapsed = Dates.now() - started
        println("Train trees:")
        print(models)

        unique_post_pruning_args = (post_pruning_args isa NamedTuple)
        if unique_post_pruning_args
            post_pruning_args = [post_pruning_args]
        end

        ret = Vector(undef, length(post_pruning_args))

        # Obtain all pruned trees
        println("Pruning $(length(post_pruning_args)) times...")
        for (i_post_pruning_args, this_pruning_args) in enumerate(post_pruning_args)

            if !(length(post_pruning_args) == 1 && this_pruning_args == (;))
                println(" Post-pruning arguments:")
                println(this_pruning_args)
            end

            this_model_tests = ModalDecisionTrees.RootLevelNeuroSymbolicHybrid[]
            this_m_trains = NamedTuple[]
            this_ms = NamedTuple[]
            this_modelts = Dates.Millisecond[]
            this_hashes = String[]

            for (i_fold, (cv_dataset_slice, cv_base_functional_model)) in enumerate(cv_base_functional_models)

                started = Dates.now()
                models_pruned = [prune(this_fold_model; this_pruning_args...) for this_fold_model in models[:, i_fold]]
                models_prunedt = Dates.now() - started
                push!(this_modelts, models_prunedt)

                model = ModalDecisionTrees.RootLevelNeuroSymbolicHybrid(cv_base_functional_model, models_pruned)

                model_save_path = ""
                hash = get_hash_sha256(model)
                push!(this_hashes, hash)
                if !isnothing(model_savedir)
                    model_save_path = model_savedir * "/nsdt_" * hash * ".jld"
                    mkpath(dirname(model_save_path))

                    checkpoint_stdout("Saving nsdt to file $(model_save_path)...")
                    JLD2.@save model_save_path model
                end

                m_train = compute_metrics(Y_train, apply_model(model, X_train))
                push!(this_m_trains, m_train)

                println(" # test instances = $(ninstances(X_test))")

                preds, model_test = sprinkle(model, X_test, Y_test)
                push!(this_model_tests, model_test)
                m = compute_metrics(Y_test, preds)
                push!(this_ms, m)

                # If not full-training
                if sort(dataset_split[1]) != sort(dataset_split[2])
                    println("Test tree:")
                    printmodel(model_test)
                end

                m_str = begin
                    if is_regression_problem
                        println()
                        display(m)
                    else
                        println(m.cm)
                        display_cm_as_row(m.cm)
                    end
                end

                cm_str = begin
                    if is_regression_problem
                        ""
                    else
                        io = IOBuffer()
                        println(io, m.cm)
                        String(take!(io))
                    end
                end

                println(cm_str * "RESULT:\t$(run_name)\t$(slice_id)\t$(this_args)\t$(modal_args)\t{$(this_pruning_args)}\t$(this_nsdt_training_args)\t$(this_nsdt_finetuning_args)\t|\t$(m_str)\t$(model_save_path)")
            end

            ret[i_post_pruning_args] = (
                this_model_tests,
                this_m_trains,
                this_ms,
                this_modelts,
                this_hashes,
            )
        end

        if unique_post_pruning_args
            ret[1]
        else
            ret
        end
    end


    function _go_nsdt_leaflevel(slicemap, X_train, Y_train, X_test, Y_test, this_args, rng; post_pruning_args, this_nsdt_training_args, this_nsdt_finetuning_args)
        slice_id, dataset_split = slicemap
        rng1 = spawn(rng)
        rng2 = spawn(rng)
        rng3 = spawn(rng)
        rng4 = spawn(rng)

        @assert !isnothing(n_nsdt_folds) "Please specificy n_nsdt_folds!"

        cv_dataset_slices = balanced_cv_dataset_slices((X_train, Y_train), n_nsdt_folds, rng1; strict=false)
        println("cv_dataset_slices: $(cv_dataset_slices)")

        # Train neural networks
        cv_base_functional_models = Dict{DatasetSplit,FunctionalModels.FunctionalModel}([
            begin
                training_idxs, validation_idxs = cv_dataset_slice
                (X_train_t, Y_train_t) = slicedataset((X_train, Y_train), training_idxs)
                (X_train_v, Y_train_v) = slicedataset((X_train, Y_train), validation_idxs)
                cv_base_functional_model = FunctionalModels.train_model((X_train_t, Y_train_t), (X_train_v, Y_train_v), class_names, rng2; this_nsdt_training_args...)
                cv_dataset_slice => cv_base_functional_model
                # (cv_dataset_slice => @cachefast "nn" model_savedir ((X_train_t, Y_train_t), (X_train_v, Y_train_v), class_names, rng2) this_nsdt_training_args FunctionalModels.train_model)
            end for cv_dataset_slice in cv_dataset_slices
        ])

        started = Dates.now()
        model =
            if timing_mode == :none
                build_tree(X_train, Y_train; this_args..., modal_args..., rng=rng3)
                # elseif timing_mode == :profile
                # @profile build_tree(X_train, Y_train; this_args..., modal_args..., rng = rng3)
            elseif timing_mode == :time
                @time build_tree(X_train, Y_train; this_args..., modal_args..., rng=rng3)
            elseif timing_mode == :btime
                @btime build_tree($X_train, $Y_train; $this_args..., $modal_args..., rng=$rng3)
            end
        time_elapsed = Dates.now() - started
        println("Train tree:")
        print(model)

        unique_post_pruning_args = (post_pruning_args isa NamedTuple)
        if unique_post_pruning_args
            post_pruning_args = [post_pruning_args]
        end

        # Obtain all pruned trees
        println("Pruning $(length(post_pruning_args)) times...")
        all_models = [
            begin
                if !(length(post_pruning_args) == 1 && this_pruning_args == (;))
                    println(" Post-pruning arguments:")
                    println(this_pruning_args)
                end
                started = Dates.now()
                model_pruned = prune(model; this_pruning_args...)
                model_prunedt = Dates.now() - started

                model_pruned_str = "$(model_pruned)"
                ((i_post_pruning_args, model_pruned_str) => (model_pruned, model_prunedt))
            end for (i_post_pruning_args, this_pruning_args) in enumerate(post_pruning_args)
        ]

        # TODO relying on string conversion for this to work. A propr way to do this would be to define something like ==(DTree, DTree)
        # Obtain all *different* pruned trees, perform train_functional_leaves and test them
        all_models_pruned_str = map((x) -> x.first[2], (all_models))
        all_models_pruned = map((x) -> x.second[1], (all_models))
        all_models_prunedt = map((x) -> x.second[2], (all_models))

        different_model_strs = unique(all_models_pruned_str)

        println("Different models produced from $(length(post_pruning_args)) parametrizations: $(length(different_model_strs))")

        ret = Vector(undef, length(post_pruning_args))
        result_strs = Vector(undef, length(post_pruning_args))

        for (i_different_model, different_model_pruned_str) in enumerate(different_model_strs)
            idxs_post_pruning_args = findall((x) -> x == different_model_pruned_str, all_models_pruned_str)
            @assert length(idxs_post_pruning_args) > 0 "Coding error.\n$(all_models_pruned_str)\n$(different_model_pruned_str)"
            model_pruned = all_models_pruned[idxs_post_pruning_args[1]]

            this_pruning_argss = post_pruning_args[idxs_post_pruning_args]
            model_prunedt = Millisecond(round(mean(all_models_prunedt[idxs_post_pruning_args] ./ Unitful.ms)) * Unitful.ms)

            println("#######################################################################")
            println(" (Different) Pruned tree $(i_different_model) / $(length(different_model_strs)):")
            println("#######################################################################")
            println("this_args = $(this_args)")
            println("pruning_argss = $(this_pruning_argss)")
            println("#######################################################################")
            println("idxs_post_pruning_args = $(idxs_post_pruning_args)")
            println("model_prunedt = $(model_prunedt)")
            println("#######################################################################")
            println(model_pruned)
            println("#######################################################################")

            this_model_enhanced_tests = DTree[]
            this_m_trains = NamedTuple[]
            this_ms = NamedTuple[]
            this_model_enhancedts = Dates.Millisecond[]
            this_hashes = String[]

            this_result_strs = String[]

            function finetuning_callback(model)
                return (datasets) -> begin
                    (_X_train_t, _Y_train_t) = datasets[1]
                    (_X_train_v, _Y_train_v) = datasets[2]
                    FunctionalModels.finetune_model(model, (_X_train_t, _Y_train_t), (_X_train_v, _Y_train_v), class_names, rng4; this_nsdt_finetuning_args...)
                end
            end

            for (i_fold, (cv_dataset_slice, cv_base_functional_model)) in enumerate(cv_base_functional_models)
                started = Dates.now()

                training_idxs, validation_idxs = cv_dataset_slice
                (X_train_t, Y_train_t) = slicedataset((X_train, Y_train), training_idxs)
                (X_train_v, Y_train_v) = slicedataset((X_train, Y_train), validation_idxs)

                model = ModalDecisionTrees.train_functional_leaves(
                    model_pruned,
                    Tuple{GenericDataset,AbstractVector}[(X_train_t, Y_train_t), (X_train_v, Y_train_v)];
                    train_callback=finetuning_callback(cv_base_functional_model),
                )
                this_model_enhancedt = Dates.now() - started
                push!(this_model_enhancedts, this_model_enhancedt)

                model_save_path = ""
                hash = get_hash_sha256(model)
                push!(this_hashes, hash)
                if !isnothing(model_savedir)
                    model_save_path = model_savedir * "/model_" * hash * ".jld"
                    mkpath(dirname(model_save_path))

                    checkpoint_stdout("Saving nsdt to file $(model_save_path)...")
                    JLD2.@save model_save_path model
                end

                m_train = compute_metrics(Y_train, apply_model(model, X_train))
                push!(this_m_trains, m_train)

                println(" # test instances = $(ninstances(X_test))")

                preds, model_enhanced_test = sprinkle(model, X_test, Y_test)
                push!(this_model_enhanced_tests, model_enhanced_test)
                m = compute_metrics(Y_test, preds)
                push!(this_ms, m)

                # If not full-training
                if sort(dataset_split[1]) != sort(dataset_split[2])
                    println("Test tree:")
                    ModalDecisionTrees.printmodel(model_enhanced_test)
                end

                m_str = begin
                    if is_regression_problem
                        println()
                        display(m)
                    else
                        display_cm_as_row(m.cm)
                    end
                end

                cm_str = begin
                    if is_regression_problem
                        ""
                    else
                        io = IOBuffer()
                        println(io, m.cm)
                        String(take!(io))
                    end
                end

                result_str = cm_str * "RESULT:\t$(run_name)\t$(slice_id)\t$(this_args)\t$(modal_args)\t{$(this_pruning_argss)}\t$(this_nsdt_training_args)\t$(this_nsdt_finetuning_args)\t|\t$(m_str)\t$(model_save_path)"
                push!(this_result_strs, result_str)
            end

            # println("nodes: ($(nnodes(model_enhanced)), height: $(height(model_enhanced)))")
            for i_post_pruning_args in idxs_post_pruning_args
                result_strs[i_post_pruning_args] = this_result_strs
                ret[i_post_pruning_args] = (
                    this_model_enhanced_tests,
                    this_m_trains,
                    this_ms,
                    (i_post_pruning_args == 1 ? time_elapsed .+ model_prunedt .+ this_model_enhancedts : model_prunedt .+ this_model_enhancedts),
                    this_hashes
                )
            end
        end

        println("Family of results:")
        println.(collect(Iterators.flatten(result_strs)))

        if unique_post_pruning_args
            ret[1]
        else
            ret
        end
    end

    function go_forest(slicemap, X_train, Y_train, X_test, Y_test, this_args, rng; post_pruning_args)
        slice_id, dataset_split = slicemap
        rng1 = spawn(rng)
        rng2 = spawn(rng)
        time_elapseds = Dates.Millisecond[]
        models = [
            begin
                started = Dates.now()
                model = begin
                    if timing_mode == :none
                        build_forest(X_train, Y_train; this_args..., modal_args..., rng=rng1)
                        # elseif timing_mode == :profile
                        # @profile build_forest(X_train, Y_train; this_args..., modal_args..., rng = rng1);
                    elseif timing_mode == :time
                        @time build_forest(X_train, Y_train; this_args..., modal_args..., rng=rng1)
                    elseif timing_mode == :btime
                        @btime build_forest($X_train, $Y_train; $this_args..., $modal_args..., rng=$rng1)
                    end
                end
                time_elapsed = (Dates.now() - started)
                push!(time_elapseds, time_elapsed)
                model
            end for i in 1:n_forest_runs]

        checkpoint_stdout("Train forests:")
        for (i, model) in enumerate(models)
            println(i)
            print(model)
        end

        unique_post_pruning_args = (post_pruning_args isa NamedTuple)
        if unique_post_pruning_args
            post_pruning_args = [post_pruning_args]
        end

        ret = [
            begin
                if !(length(post_pruning_args) == 1 && this_pruning_args == (;))
                    println(" Post-pruning arguments:")
                    println(this_pruning_args)
                end

                this_model_pruneds = DForest[]
                this_m_trains = NamedTuple[]
                this_ms = NamedTuple[]
                this_model_prunedts = Dates.Millisecond[]
                this_hashes = String[]

                for model in models
                    started = Dates.now()
                    model_pruned = prune(model, rng=rng2; this_pruning_args...)
                    push!(this_model_pruneds, model_pruned) # TODO save test models
                    this_model_prunedt = Dates.now() - started
                    push!(this_model_prunedts, this_model_prunedt)

                    println(" Pruned forest $(i_post_pruning_args) / $(length(post_pruning_args)):")
                    println(model_pruned)

                    model_save_path = ""
                    hash = get_hash_sha256(model_pruned)
                    push!(this_hashes, hash)
                    if !isnothing(model_savedir)
                        model_save_path = model_savedir * "/model_" * hash * ".jld"
                        mkpath(dirname(model_save_path))

                        checkpoint_stdout("Saving forest to file $(model_save_path)...")
                        JLD2.@save model_save_path model_pruned
                    end

                    m_train = compute_metrics(Y_train, apply_model(model_pruned, X_train))
                    push!(this_m_trains, m_train)

                    println(" # test instances = $(ninstances(X_test))")

                    preds = apply_model(model_pruned, X_test)
                    m = compute_metrics(Y_test, preds)
                    push!(this_ms, m)

                    m_str = begin
                        if is_regression_problem
                            println()
                            display(m)
                        else
                            println(m.cm)
                            display_cm_as_row(m.cm)
                        end
                    end

                    println("RESULT:\t$(run_name)\t$(slice_id)\t$(this_args)\t$(modal_args)\t$(this_pruning_args)\t|\t$(m_str)\t$(model_save_path)")
                end

                (this_model_pruneds, this_m_trains, this_ms, (i_post_pruning_args == 1 ? time_elapseds .+ this_model_prunedts : this_model_prunedts), this_hashes)
            end for (i_post_pruning_args, this_pruning_args) in enumerate(post_pruning_args)]

        if unique_post_pruning_args
            ret[1]
        else
            ret
        end
    end

    go(slicemap::Tuple{Any,AbstractDatasetSplit}, dataset_filepath, X_train, Y_train, X_test, Y_test) = begin

        (slice_id, dataset_split) = slicemap

        tree_results = []
        nsdt_results = []
        forest_results = []

        # TREE

        optimize_tree_computation = true
        # Debug: tree_args = Random.shuffle(tree_args)[1:4]
        nondominated_pars, ndp_perm = ModalDecisionTrees.nondominated_pruning_parametrizations(tree_args, do_it_or_not=optimize_tree_computation, return_perm=true)
        # println(tree_args)
        # println(nondominated_pars, ndp_perm)
        if length(nondominated_pars) > 0
            println("tree ndp_perm: $(ndp_perm)")
        end
        for (i_model, (this_args, post_pruning_args)) in enumerate(nondominated_pars)
            checkpoint_stdout("Computing master tree $(i_model) / $(length(nondominated_pars))...\n$(this_args)")
            push!(tree_results, go_tree(
                slicemap,
                X_train,
                Y_train,
                X_test,
                Y_test,
                this_args,
                Random.MersenneTwister(train_seed);
                post_pruning_args=post_pruning_args,
            )
            )
        end
        tree_results = [tree_results[i][j] for (i, j) in ndp_perm]

        # NSDT

        optimize_nsdt_computation = true

        nondominated_pars, ndp_perm = ModalDecisionTrees.nondominated_pruning_parametrizations(
            nsdt_args,
            do_it_or_not=optimize_nsdt_computation,
            return_perm=true,
            ignore_additional_args=[:hybrid_type],
        )
        if length(nondominated_pars) > 0
            println("nsdt ndp_perm: $(ndp_perm)")
        end
        nsdt_results = []
        for (i_this_nsdt_training_args, this_nsdt_training_args) in enumerate(nsdt_training_args)
            for (i_this_nsdt_finetuning_args, this_nsdt_finetuning_args) in enumerate(nsdt_finetuning_args)
                this_nsdt_results = [
                    begin
                        checkpoint_stdout("Computing master nsdt $((i_this_nsdt_training_args,i_this_nsdt_finetuning_args,i_model)) " *
                                          "/ $(length.((nsdt_training_args, nsdt_finetuning_args, nondominated_pars)))...\n$(this_args)")
                        go_nsdt(
                            slicemap,
                            X_train,
                            Y_train,
                            X_test,
                            Y_test,
                            this_args,
                            Random.MersenneTwister(train_seed);
                            post_pruning_args=post_pruning_args,
                            this_nsdt_training_args=this_nsdt_training_args,
                            this_nsdt_finetuning_args=this_nsdt_finetuning_args,
                        )
                    end for (i_model, (this_args, post_pruning_args)) in enumerate(nondominated_pars)
                ]
                [push!(nsdt_results, this_nsdt_results[i][j]) for (i, j) in ndp_perm]
            end
        end

        # FOREST

        nondominated_pars, ndp_perm = ModalDecisionTrees.nondominated_pruning_parametrizations(forest_args, do_it_or_not=optimize_forest_computation, return_perm=true)
        if length(nondominated_pars) > 0
            println("forest ndp_perm: $(ndp_perm)")
        end
        for (i_model, (this_args, post_pruning_args)) in enumerate(nondominated_pars)
            checkpoint_stdout("Computing master forest $(i_model) / $(length(nondominated_pars))...\n$(this_args)")
            push!(forest_results, go_forest(
                slicemap,
                X_train,
                Y_train,
                X_test,
                Y_test,
                this_args,
                Random.MersenneTwister(train_seed);
                post_pruning_args=post_pruning_args,
            )
            )
        end
        forest_results = [forest_results[i][j] for (i, j) in ndp_perm]

        ##############################################################################
        ##############################################################################
        # PRINT RESULT IN FILES
        ##############################################################################
        ##############################################################################

        # PRINT FULL
        # println(typeof(X_train))
        # println(typeof(X_test))
        # println(length(Y_train))
        # println(typeof(Y_train))
        # println(length(Y_test))
        # println(typeof(Y_test))
        # println(ninstances(X_train))
        # println(ninstances(X_test))

        dataset_slice = (dataset_split[1], dataset_split[2])

        values_for_condition_columns = [
            slice_id,
            [replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)]...,
            [
                begin
                    other_function = other_functions[col_name]
                    string(other_function(X_train, Y_train, X_test, Y_test, dataset_slice, dataset_filepath))
                end for col_name in other_columns
            ]...,
        ]

        models_results_n_columns = [
            (tree_results, tree_columns),
            (nsdt_results, nsdt_columns),
            (forest_results, forest_columns),
        ]

        print_result_row(
            full_output_filepath,
            models_results_n_columns,
            values_for_condition_columns,
            model_savedir,
            results_col_sep,
        )

        callback(slice_id)

        Dict(
            "tree_results" => tree_results,
            "nsdt_results" => nsdt_results,
            "forest_results" => forest_results,
        )
    end

    ##############################################################################
    ##############################################################################
    ##############################################################################

    # println(typeof(dataset_slices))

    if isa(dataset_slices, AbstractVector{<:AbstractDatasetSlice}) || isa(dataset_slices, AbstractVector{<:AbstractDatasetSplit})
        dataset_slices = enumerate(dataset_slices) |> collect
    elseif isnothing(dataset_slices)
        dataset_slices = [(0, nothing)]
    end

    # println(typeof(dataset_slices))

    # TODO -> make it AbstractVector{<:AbstractDatasetSplit}
    # if isa(dataset_slices,AbstractVector{<:AbstractDatasetSlice})
    # isa(dataset_slices,AbstractVector{<:AbstractDatasetSlice})
    # end


    @assert isa(dataset_slices, AbstractVector{<:Tuple{<:Any,Union{Nothing,<:AbstractDatasetSplitOrSlice}}}) "$(dataset_slices)\n$(typeof(dataset_slices))"
    # @assert isa(dataset_slices, AbstractVector{<:Tuple{<:Any,<:AbstractDatasetSplit}}) "$(dataset_slices)\n$(typeof(dataset_slices))"

    ##############################################################################
    ##############################################################################
    ##############################################################################

    println()
    println("train_seed   = ", train_seed)
    println("modal_args   = ", modal_args)
    println("tree_args    = ", tree_args)
    println("nsdt_args    = ", nsdt_args)
    println("n_nsdt_folds = ", n_nsdt_folds)
    println("nsdt_training_args = ", nsdt_training_args)
    println("nsdt_finetuning_args = ", nsdt_finetuning_args)
    println("forest_args  = ", forest_args)
    println("n_forest_runs  = ", n_forest_runs)
    # println("forest_args  = ", length(forest_args), " × some forest_args structure")
    println()
    println("split_threshold   = ", split_threshold)
    println("data_modal_args   = ", data_modal_args)
    println("dataset_slices    = ($(length(dataset_slices)) dataset_slices)")

    # println("round_dataset_to_datatype   = ", round_dataset_to_datatype)
    # println("use_training_form   = ", use_training_form)
    # println("data_savedir   = ", data_savedir)
    # println("model_savedir   = ", model_savedir)
    # println("log_level   = ", log_level)
    # println("timing_mode   = ", timing_mode)

    println()

    old_logger = global_logger(logger)

    rets = []

    makelogisets_fun = (X_train, X_test) -> makelogisets(X_train, X_test, data_modal_args, modal_args, use_training_form, data_savedir, timing_mode, save_datasets, use_test_form)

    X_full = nothing
    X_full_input = nothing

    for (i_slice, (X_f_tuple, slice_id, ((X_train, Y_train), (X_test, Y_test)), dataset_split)) in enumerate(generate_splits(dataset, split_threshold, round_dataset_to_datatype, save_datasets, run_name, dataset_slices, data_savedir, makelogisets_fun))

        (X_full, X_full_input) = X_f_tuple

        @assert ninstances(X_train) == length(Y_train) "$(typeof(X_train)). $(ninstances(X_train)) != $(length(Y_train))"
        @assert ninstances(X_test) == length(Y_test) "$(typeof(X_test)). $(ninstances(X_test)) != $(length(Y_test))"


        println("train dataset:")
        println("  " * displaystructure(X_train; indent_str="  "))

        println("test  dataset:")
        println("  " * displaystructure(X_test; indent_str="  "))

        if !skip_training
            if resilient_mode
                try
                    push!(rets, go((slice_id, dataset_split), dataset_filepath, X_train, Y_train, X_test, Y_test))
                catch e
                    println("exec_run: An error occurred!")
                    println(e)
                    return
                end
            else
                push!(rets, go((slice_id, dataset_split), dataset_filepath, X_train, Y_train, X_test, Y_test))
            end
        end
    end

    # Finally save the dataset with memoization
    if save_datasets && use_training_form == :supportedlogiset_with_memoization && !isnothing(X_full)
        @cachefast_overwrite "training_dataset" data_savedir ("train", data_modal_args, X_full_input, modal_args, save_datasets, timing_mode, data_savedir, use_training_form) makelogiset X_full
    end

    global_logger(old_logger)

    # Iterators.flatten(first(rets)) |> collect
    # rets = zip(rets...) |> collect
    # rets = map((r)->Iterators.flatten(r) |> collect, rets)
    # rets

    # rets=[a,b] # debug
    ks = unique(Iterators.flatten(keys.(rets)) |> collect)
    all_rets = Dict()
    for k in ks
        all_rets[k] = [
            if haskey(r, k)
                r[k]
            else
                nothing
            end
            for r in rets]
    end
    all_rets
end
