################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################
DEBUG = false
# DEBUG = true

if DEBUG
    dataset_origin = "Mozilla"
    file_name = "respiratory_Healthy_Pneumonia"
    file_path = "/home/paso/datasets/Speech/respiratory"

    op_label = "propositional"
    op_label = "modal"
    op_label = "multimodal"
else
    println("Arguments recieved:")
    println(ARGS[1])
    println(ARGS[2])
    println(ARGS[3])
    println(ARGS[4])

    dataset_origin = ARGS[1]
    file_name = ARGS[2]
    file_path = ARGS[3]
    op_label = ARGS[4]
end

####### whitout these 2 pkg, jld2 won't load ########
using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end
# using Arrow
using CSV

using Catch22
using DataFrames
using JLD2
using DataStructures

include("scanner.jl")
include("datasets/dataset-analysis.jl")
include("docs/all.jl")

train_seed = 1

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = (string("speech/gender_age_results/", file_name, "_", op_label))
# results_dir = (string("speech/gap/results/", file_name))
# results_dir = (string("speech/sspeech_results-abandonment-audio_end-temporal/", file_name))

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

dry_run = false
# dry_run = :dataset_only
# dry_run = :model_study
# dry_run = true

skip_training = false
# skip_training = true

#save_datasets = true
save_datasets = false

perform_consistency_check = false # true

iteration_blacklist = []

use_training_form = :supportedlogiset_with_memoization

############################################################################################

# # feature_selection_method = :variance
# feature_selection_method = :pvalue

# perform_target_aware_analysis = false
# # perform_target_aware_analysis = true

# perform_feature_selection = false
# # perform_feature_selection = true

# # savefigs = true
# savefigs = false

# n_desired_variables = 5
# n_desired_descriptors = 5

############################################################################################
########################################### TREES ##########################################
############################################################################################

# Optimization arguments for single-tree
tree_args = [
#   (
#       loss_function = nothing, # ModalDecisionTrees.entropy
#       min_samples_leaf = 1,
#       min_purity_increase = 0.01,
#       max_purity_at_leaf = 0.6,
#   )
]

for loss_function in [nothing] # ModalDecisionTrees.variance
    for min_samples_leaf in [2, 4] # [1,2]
        for min_purity_increase in [0.01, 0.05, 0.1]
            for max_purity_at_leaf in [Inf, 0.5, 0.6]
                # for max_purity_at_leaf in [10]
                push!(tree_args,
                    (
                        loss_function=loss_function,
                        min_samples_leaf=min_samples_leaf,
                        min_purity_increase=min_purity_increase,
                        max_purity_at_leaf=max_purity_at_leaf,
                        perform_consistency_check=perform_consistency_check,
                    )
                )
            end
        end
    end
end

println(" $(length(tree_args)) trees")

############################################################################################
######################################## NSDT ARGS #########################################
############################################################################################

n_nsdt_folds = 4

nsdt_args = []

for loss_function in []
    # for loss_function in [nothing]
    for min_samples_leaf in [4, 16]
        for min_purity_increase in [0.1, 0.05, 0.02, 0.01] # , 0.0075, 0.005, 0.002]
            for max_purity_at_leaf in [Inf, 0.001] # , 0.01, 0.2, 0.4, 0.6]
                push!(nsdt_args,
                    (;
                        loss_function=loss_function,
                        min_samples_leaf=min_samples_leaf,
                        min_purity_increase=min_purity_increase,
                        max_purity_at_leaf=max_purity_at_leaf,
                        perform_consistency_check=perform_consistency_check
                    )
                )
            end
        end
    end
end


nsdt_training_args = []
for model_type in [:lstm] # [:lstm, :tran]
    for epochs in [100] #
        for batch_size in [32] # 64 # 16
            for hidden_size in [64] #
                for code_size in [1] #
                    for save in [false] #
                        for use_subseries in [false]
                            for learning_rate in [1e-5] # 5e-5 # 1e-4
                                for patience in [nothing] # , nothing]
                                    push!(nsdt_training_args,
                                        (;
                                            model_type=model_type,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            hidden_size=hidden_size,
                                            code_size=code_size,
                                            save=save,
                                            use_subseries=use_subseries,
                                            learning_rate=learning_rate,
                                            patience=patience
                                        )
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


nsdt_finetuning_args = []
for epochs in [10]
    for batch_size in [4] # 2
        for save in [false]
            for use_subseries in [false]
                for learning_rate in [5e-6] # 5e-5
                    for patience in [5] # , nothing]
                        push!(nsdt_finetuning_args,
                            (;
                                epochs=epochs,
                                batch_size=batch_size,
                                save=save,
                                use_subseries=use_subseries,
                                learning_rate=learning_rate,
                                patience=patience
                            )
                        )
                    end
                end
            end
        end
    end
end

println(" $(length(nsdt_args)) nsdts " * (length(nsdt_args) > 0 ? "(with $(n_nsdt_folds) folds)" : ""))

############################################################################################
########################################## FORESTS #########################################
############################################################################################

n_forest_runs = 3
optimize_forest_computation = true


forest_args = []

# for ntrees in []
for ntrees in [100]
    for n_subfeatures in [half_f]
        for n_subrelations in [id_f]
            for partial_sampling in [0.7]
                push!(forest_args, (
                    n_subfeatures=n_subfeatures,
                    ntrees=ntrees,
                    partial_sampling=partial_sampling,
                    n_subrelations=n_subrelations,
                    # Optimization arguments for trees in a forest (no pruning is performed)
                    loss_function=nothing, # ModalDecisionTrees.entropy
                    # min_samples_leaf    = 1,
                    # min_purity_increase = ...,
                    # max_purity_at_leaf  = ..,
                    perform_consistency_check=perform_consistency_check,
                ))
            end
        end
    end
end

println(" $(length(forest_args)) forests " * (length(forest_args) > 0 ? "(repeated $(n_forest_runs) times)" : ""))

################################################################################
################################## MODAL ARGS ##################################
################################################################################

modal_args = (;
    initconditions=ModalDecisionTrees.start_without_world,
    # initconditions = ModalDecisionTrees.start_at_center,
    # allow_global_splits = true,
    allow_global_splits=false
)

data_modal_args = (;
    # relations=[globalrel, SoleLogics.IARelations...]
    relations=[globalrel, SoleLogics.IA7Relations...]
    #     relations=SoleLogics.AbstractRelation[]
    # mixed_conditions   = [canonical_geq_80, canonical_leq_80], ### DA PROVARE
)

################################################################################
############################### FEATURE SELECTION ##############################
################################################################################

# feature_selection_method = :feature
# feature_selection_method = :variable
feature_selection_method = :none

feature_selectors = [
    ( # STEP 1: unsupervised variance-based filter
        selector=SoleFeatures.VarianceFilter(SoleFeatures.IdentityLimiter()),
        limiter=PercentageLimiter(0.5),
    ),
    ( # STEP 2: supervised Mutual Information filter
        selector=SoleFeatures.MutualInformationClassif(SoleFeatures.IdentityLimiter()),
        limiter=PercentageLimiter(0.1),
    ),
    ( # STEP 3: group results by variable
        selector=IdentityFilter(),
        limiter=SoleFeatures.IdentityLimiter(),
    ),
]

feature_selection_aggrby = nothing

feature_selection_validation_n_test = 10
feature_selection_validation_seed = 5

################################################################################
##################################### MISC #####################################
################################################################################

# log_level = Logging.Warn
# log_level = SoleBase.LogOverview
# log_level = SoleBase.LogDebug
log_level = SoleBase.LogDetail

# timing_mode = :none
timing_mode = :time
# timing_mode = :btime
#timing_mode = :profile

round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8
# round_dataset_to_datatype = UInt16
# round_dataset_to_datatype = UInt32
# round_dataset_to_datatype = UInt64
# round_dataset_to_datatype = Float16
# round_dataset_to_datatype = Float32
# round_dataset_to_datatype = Float64

traintest_threshold = 1.0

# split_threshold = 0.0
split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = [(1:10)...]

exec_dataset_origin = [
    dataset_origin
]

exec_dataset_name = [
    ((string(file_path, "/", file_name)), op_label)
]

exec_moving_average_args = [
    (
        nwindows=5,
        relative_overlap=0.2,
    ),
    # (
    #     nwindows=10,
    #     relative_overlap=0.2,
    # ),
    # (
    #     nwindows=15,
    #     relative_overlap=0.2,
    # ),
]

# op_mode = Dict(
#     "propositional" => "canonical_geq",
#     "modal" => "canonical",
# )

if op_label == "propositional"

    exec_multimodal_modes = [
        [:static],
        # [:modal],
        # [:modal, :static],
        # [:static_allintervals],
        # [:modal, :static_allintervals],
    ]

    exec_mixed_conditions = [
        # op_mode[op_label],
        # nothing,
        # "canonical",
        "catch9",
        # "Catch22",
        # MixedCondition[minimum, maximum, StatsBase.mean],
        # "canonical_80",
        # "Mean",
    ]

elseif op_label == "modal"

    exec_multimodal_modes = [
        # [:static],
        [:modal],
        # [:modal, :static],
        # [:static_allintervals],
        # [:modal, :static_allintervals],
    ]

    exec_mixed_conditions = [
        "canonical",
    ]

elseif op_label == "multimodal"

    exec_multimodal_modes = [
        # [:static],
        # [:modal],
        [:modal, :static],
        # [:static_allintervals],
        # [:modal, :static_allintervals],
    ]

    exec_mixed_conditions = [
        ["canonical", "catch9"],
    ]

else
    error("Unknown op_label: $op_label")
end


exec_ranges = (;
    dataset_origin=exec_dataset_origin,
    dataset_name=exec_dataset_name,
    moving_average_args_params=exec_moving_average_args,
    mixed_conditions=exec_mixed_conditions,
    multimodal_modes=exec_multimodal_modes
)


dataset_function = (dataset_origin, dataset_name, moving_average_args_params) -> begin

    (dataset_name, y) = dataset_name

    @show dataset_name

    if dataset_origin == "Mozilla" || dataset_origin == "Ravdess"
        # Note: Requires Catch22
        d = jldopen(string(dataset_name, ".jld2"))
        # df, Y, vars_n_meas = d["dataframe_validated"]
        df, Y = d["dataframe_validated"]

        @assert df isa DataFrame

        X, variable_names = SoleData.dataframe2cube(df)

        X = moving_average_filter(X; moving_average_args_params...)

        # Y = string.(Y)
        # Y = Y[:, 1] # standard Mozilla 
        # # Y = Y[!, label]
        # Y = Vector{String}(Y)

        dataset = (X, Y)

        # @show typeof(vars_n_meas)
        # @show vars_n_meas
        # vars, meas = vars_n_meas
        # vars = map(a->parse(Int, string(a)[2:end]), vars)
        # vars_n_meas = collect(zip(vars, meas))

        return dataset, variable_names
    elseif dataset_origin == "GAP"
        if y != "abandonment"
            throw(ArgumentError("Currently only `abandonment` label is supported for GAP dataset; passed: $y"))
        end

        statrfrom = :end
        # TODO: more parameters from outside!
        df = @scache_if !isnothing(data_savedir) "gap_dataset" data_savedir loaddatasetGAP(
            joinpath(ENV["DATA_DIR"], dataset_origin, dataset_name),
            joinpath(ENV["DATA_DIR"], dataset_origin, string(dirname(dataset_name), "_partitioned"));
            nsamples=:all,
            startfrom=statrfrom,
            rng=Random.default_rng(), # FIXME: this can't use default_rng for reproducibility
            concat=true,
        )

        # discard all audio with less than 25_000 samples
        to_keep = findall(samples -> length(samples) ≥ 25_000, df[:, :audio])
        df = df[to_keep, :]

        # TODO: more parameters from outside!
        extracted_features, variable_names = zip(mfcc_extended.(df[:, :audio], df[1, :sampling_rate])...)
        variable_names = first(variable_names)

        nfeats = length(eachrow(first(extracted_features)))

        insertcols!(df, [variable_names[i_feat] => getindex.([eachrow.(extracted_features)...], i_feat) for i_feat in 1:nfeats]...)

        # cut to smallest
        df[!, variable_names] = cuttosmallest!(deepcopy(df[:, variable_names]); from=statrfrom)

        X, variable_names = SoleData.dataframe2cube(df[:, variable_names])

        # TODO: enable this before performing learning
        # X = moving_average_filter(X; moving_average_args_params...)

        Y =
            if y == "abandonment"
                replace(df[:, y],
                    missing => "data_collection_not_completed",
                    "info_endpoint" => "data_collection_not_completed"
                )
            else
                Y
            end

        dataset = (X, Y)

        return dataset, variable_names
    else
        throw(ArgumentError("Unsupported dataset origin: $dataset_origin"))
    end
end

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

iteration_whitelist = []


############################################################################################
############################################################################################
############################################################################################
################################################################################

models_to_study = Dict([
# (
# parametri 4 prime 4 colonne del full colum (tranne dataseed)
#   "fcmel",8000,false,"stump_with_memoization",("c",3,true,"KDD-norm-partitioned-v1",["NG","Normalize","RemSilence"]),30,(max_points = 50, ma_size = 30, ma_step = 20),false,"TestOp_80","IA"
# ) => [
#   "tree_d3377114b972e5806a9e0631d02a5b9803c1e81d6cd6633b3dab4d9e22151969"
# modelli di mozilla
# ],
])

models_to_study = Dict(JSON.json(k) => v for (k, v) in models_to_study)


############################################################################################
############################################################################################
############################################################################################
################################################################################

mkpath(results_dir)

if "-f" in ARGS
    if isfile(iteration_progress_json_file_path)
        println("Backing up existing $(iteration_progress_json_file_path)...")
        backup_file_using_creation_date(iteration_progress_json_file_path)
    end
end

# Copy scan script into the results folder
if PROGRAM_FILE != ""
    backup_file_using_creation_date(PROGRAM_FILE; copy_or_move=:copy, out_path=results_dir)
end

exec_ranges_names, exec_ranges_iterators = collect(string.(keys(exec_ranges))), collect(values(exec_ranges))
history = load_or_create_history(
    iteration_progress_json_file_path, exec_ranges_names, exec_ranges_iterators
)

############################################################################################
############################################################################################
############################################################################################
################################################################################

# Log to console AND to .out file, & send Telegram message with Errors
using Logging, LoggingExtras
using Telegram, Telegram.API
using ConfigEnv

i_log_filename, log_filename = 0, ""
while i_log_filename == 0 || isfile(log_filename)
    global i_log_filename, log_filename
    i_log_filename += 1
    log_filename =
        results_dir * "/" *
        (dry_run == :dataset_only ? "datasets-" : "") *
        "$(i_log_filename).out"
end
logfile_io = open(log_filename, "w+")
dotenv()

if haskey(ENV, "TELEGRAM_BOT_TOKEN") && haskey(ENV, "TELEGRAM_BOT_CHAT_ID")
    tg = TelegramClient()
    tg_logger = TelegramLogger(tg; async=false)

    new_logger = TeeLogger(
        current_logger(),
        SimpleLogger(logfile_io, log_level),
        MinLevelLogger(tg_logger, Logging.Error), # Want to ignore Telegram? Comment out this
    )
    global_logger(new_logger)
end

# TODO actually,no need to recreate the dataset when changing, say, testoperators. Make a distinction between dataset params and run params
n_interations = 0
n_interations_done = 0
for params_combination in collect(IterTools.product(exec_ranges_iterators...))

    flush(logfile_io)

    # Unpack params combination
    # params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)
    params_namedtuple = (; zip(Symbol.(exec_ranges_names), params_combination)...)

    # FILTER ITERATIONS
    if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
        continue
    end

    global n_interations += 1

    ##############################################################################
    ##############################################################################
    ##############################################################################

    run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

    # Placed here so we can keep track of which iteration is being skipped
    print("Iteration \"$(run_name)\"")

    # Check whether this iteration was already computed or not
    if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets)
        println(": skipping")
        continue
    else
        println("...")
    end

    global n_interations_done += 1

    if dry_run == true
        continue
    end

    ##############################################################################
    ##############################################################################
    ##############################################################################

    local dataset_origin
    (
        dataset_origin,
        dataset_name,
        moving_average_args,
        mixed_conditions,
        multimodal_modes,
    ) = params_combination

    dataset_fun_sub_params = (
        dataset_origin,
        dataset_name,
        moving_average_args,
    )

    cur_modal_args = deepcopy(modal_args)
    cur_data_modal_args = deepcopy(data_modal_args)

    if dry_run == :model_study
        # println(JSON.json(params_combination))
        # println(models_to_study)
        # println(keys(models_to_study))
        if JSON.json(params_combination) in keys(models_to_study)

            trees = models_to_study[JSON.json(params_combination)]

            println()
            println()
            println("Study models for $(params_combination): $(trees)")

            if length(trees) == 0
                continue
            end

            println("dataset_fun_sub_params: $(dataset_fun_sub_params)")

            # @assert dataset_fun_sub_params isa String

            # dataset_fun_sub_params = merge(dataset_fun_sub_params, (; mode = :testing))

            datasets = []
            println("TODO")
            # datasets = [
            #   (mode,if dataset_fun_sub_params isa Tuple
            #       dataset = dataset_function(dataset_fun_sub_params...; mode = mode)
            #       # dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
            #       (X, Y), (n_pos, n_neg) = dataset
            #       # elseif dataset_fun_sub_params isa String
            #       #   # load_cached_obj("dataset", data_savedir, dataset_fun_sub_params)
            #       #   dataset = Serialization.deserialize("$(data_savedir)/dataset_$(dataset_fun_sub_params).jld").train_n_test
            #       #   println(typeof(dataset))
            #       #   (X, Y), (n_pos, n_neg) = dataset
            #       #   (X, Y, nothing), (n_pos, n_neg)

            #       # TODO should not need these at test time.
            #       X = build_logiset(X, test_operators, relations)
            #       # println(length(Y))
            #       # println((n_pos, n_neg))

            #       println(X)
            #       # println(Y)
            #       dataset = (X, Y), (n_pos, n_neg)
            #       dataset
            #   else
            #       throw_n_log("$(typeof(dataset_fun_sub_params))")
            #   end) for mode in [:testing, :development]
            # ]

            for model_hash in trees

                println()
                println()
                println("Loading model: $(model_hash)...")

                model = load_model(model_hash, model_savedir)

                println()
                println("Original model (training):")
                if model isa DTree
                    printmodel(model)
                end

                for (mode, dataset) in datasets
                    (X, Y), (n_pos, n_neg) = dataset

                    println()

                    println()
                    println("Regenerated model ($(mode)):")

                    if model isa DTree
                        predictions, regenerated_model = printapply(model, X, Y)
                        println()
                        # printmodel(regenerated_model)
                    end

                    preds = apply_model(model, X)
                    cm = confusion_matrix(Y, preds)
                    println(cm)

                    # readline()
                end
            end
        end
    end

    # Load Dataset
    # TODO: is it better cache here or inside function?
    #dataset, variable_names = @scache_if !isnothing(data_savedir) "dataset" data_savedir dataset_function(dataset_fun_sub_params...)
    dataset, variable_names = dataset_function(dataset_fun_sub_params...)

    cur_data_modal_args = merge(cur_data_modal_args, (;
        mixed_conditions=begin
            if isnothing(mixed_conditions)
                vcat([
                    begin
                        [
                            (≥, SoleData.UnivariateFeature{Float64}(var, get_patched_feature(meas, :pos))),
                            (≤, SoleData.UnivariateFeature{Float64}(var, get_patched_feature(meas, :neg)))
                        ]
                    end for (var, meas) in vars_n_meas
                ]...)
            elseif mixed_conditions isa AbstractString
                mixed_conditions_dict[mixed_conditions]
            elseif mixed_conditions isa AbstractVector{<:AbstractString}
                map(x -> mixed_conditions_dict[x], mixed_conditions)
            else
                mixed_conditions
            end
        end
    ))

    ##############################################################################
    ##############################################################################
    ##############################################################################

    X, Y = dataset

    if feature_selection_method != :none
        println("Size before feature selection: $(size(X))")

        # convert to DataFrame
        X = matricial2df(X; varnames=variable_names)

        X, fs_mid_res =
            if feature_selection_method == :feature
                println("Executing Feature selection (NOTE: this will make the dataset static)")
                feature_selection(X, Y; fs_methods=feature_selectors, return_mid_results=Val(true), aggrby=feature_selection_aggrby)
            elseif feature_selection_method == :variable
                println("Executing Variable selection")
                variable_selection(X, Y; fs_methods=feature_selectors, return_mid_results=Val(true))
            else
                throw(ErrorException("Unknown feature selection method $feature_selection_method"))
            end

        #         pretty_print_feature_selection_study(study_feature_selection_mid_results(fs_mid_res))
        #
        #         random_features_subset = shuffle(MersenneTwister(feature_selection_validation_seed), 1:ncol(X))[1:feature_selection_validation_n_test]
        #         pvalues, _ = validate_features(X[:,random_features_subset], Y)
        #
        #         wrap_feature_selection(
        #             joinpath(results_dir, "fs-results.tex"), fs_mid_res;
        #             features2pvalues = OrderedDict(zip(names(X), pvalues))
        #         )

        # convert back to Matricial
        X = md2matricial(MultiDataset(X; group=:all))[1]

        println("Size after feature selection: $(size(X))")
    end

    # cur_data_modal_args = merge(cur_data_modal_args, (;
    #     mixed_conditions = Vector{MixedCondition}(
    #         collect(Iterators.flatten(getcanonicalfeatures.(best_descriptors)))
    #     )
    # ))

    # println()
    # println("cur_data_modal_args.mixed_conditions = $(cur_data_modal_args.mixed_conditions)")
    # println("new size(X) = $(size(X))")
    # println()

    dataset = ([X], Y)

    ##############################################################################
    ##############################################################################
    ##############################################################################

    Xs, Y = dataset

    mixed_conditions, relations = cur_data_modal_args.mixed_conditions, cur_data_modal_args.relations

    mixed_conditionss = begin
        if mixed_conditions isa AbstractVector{<:AbstractVector}
            mixed_conditions
        else
            [mixed_conditions]
        end
    end

    @assert length(Xs) == 1 "Loaded dataset is multimodal!!! Can't apply multimodal_modes to multimodal dataset"
    Xs_varnames = SoleData.dataframe2cube.(matricial2df.(Xs; vartype=Float64))
    Xs = [X for (X, varnames) in Xs_varnames]
    X = Xs[1]

    Xs, mixed_conditionss, relationss = apply_multimodal_modes(
        X, # AbstractArray{T,3}
        multimodal_modes, # ::Vector{Vector}
        mixed_conditionss, # ::Vector{Vector}
        relations # ::Vector
    )
    # @show typeof(Xs)
    # @show typeof.(Xs)

    # Xs = df2matricial.(SoleData.cube2dataframe.(Xs))
    Xs = cube2matricial.(Xs)
    # @show typeof(Xs)
    # @show typeof.(Xs)
    cur_data_modal_args = merge(cur_data_modal_args,
        (
            mixed_conditions=mixed_conditionss,
            relations=relationss,
        )
    )

    # if all(isa.(values(cur_data_modal_args), AbstractVector)) && all(length.(values(cur_data_modal_args)) .== _nmodalities)
    # println("Permuting cur_data_modal_args...")
    # println("$(cur_data_modal_args)")
    cur_data_modal_args = permute_named_tuple(cur_data_modal_args)
    println("-> $(cur_data_modal_args)")
    # end

    dataset = (Xs, Y)

    ##############################################################################
    ##############################################################################
    ##############################################################################

    # n_insts = length(Y)

    # println(datasource_counts)
    # train_datasource_counts = datasource_counts[1:datasource_traintest_counts[1]]
    # test_datasource_counts = datasource_counts[(datasource_traintest_counts[1]+1):end]

    # # separate_test_idxs = Vector{Integer}(sum(sum.(train_datasource_counts)):n_insts)
    # separate_train_idxs = 1:sum(sum.(train_datasource_counts))
    # separate_test_idxs = (sum(sum.(train_datasource_counts))+1):n_insts
    # @assert sum(sum.(test_datasource_counts)) == length(separate_test_idxs) "$(sum(sum.(test_datasource_counts))) != $(separate_test_idxs)"
    # println("separate_test_idxs = $(separate_test_idxs)")

    # # dataset = ([X], Y)
    # X_train, Y_train = [X[separate_train_idxs]], Y[separate_train_idxs]
    # X_test , Y_test  = [X[separate_test_idxs]] , Y[separate_test_idxs]
    # dataset = ((X_train, Y_train), (X_test, Y_test))

    # X, Y = X_train, Y_train
    # n_insts = length(Y)
    ##############################################################################
    ##############################################################################
    ##############################################################################

    _, Y = dataset
    class_count = get_class_counts(Y, true)

    # println("class_distribution: ")
    # println(StatsBase.countmap(dataset[2]))
    # println("class_counts: $(class_count)")

    ## Dataset slices

    # TODO: dataset_slices = randomized_cv_dataset_slices(dataset, 0.8, ninstances_per_class, todo_dataseeds)
    # randomized cross validation
    # dataset_slices = [(dataseed, begin
    #     split_threshold = 0.8
    #     idxs = balanced_dataset_slice(dataset, dataseed; ninstances_per_class = ninstances_per_class)
    #     train_idxs, test_idxs = idxs[1:floor(Int, length(idxs)*split_threshold)], idxs[1+floor(Int, length(idxs)*split_threshold):end]
    #     (train_idxs, test_idxs)
    # end) for dataseed in todo_dataseeds]

    # TODO: dataset_slices = kfold_cv_dataset_slices(dataset, dataseeds, todo_dataseeds)
    # k-fold cross validation
    # dataset_slices = begin
    #     n_folds = length(dataseeds)
    #     dataset_slices = balanced_cv_dataset_slices(dataset, n_folds; strict = true, rng = data_seed)
    #     slices_idxs = [findfirst((x)->x==dataseed, dataseeds) for dataseed in todo_dataseeds]
    #     dataset_slices = dataset_slices[slices_idxs]
    #     collect(zip(todo_dataseeds, dataset_slices))
    # end
    # obtain dataseeds that are were not done before
    todo_dataseeds = filter((dataseed) -> !iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

    # println(datasource_counts)
    # println(uniq(Y))

    # println(datasource_counts)

    dataset_slices = begin
        n_insts = length(Y)
        # @assert (n_insts % n_cv_folds == 0) "$(n_insts) % $(n_cv_folds) != 0"
        # n_insts_fold = div(n_insts, n_cv_folds)

        # todo_dataseeds
        Vector{Tuple{Integer,Union{DatasetSlice,Tuple{DatasetSlice,DatasetSlice}}}}([(dataseed, begin
            if dataseed == 0
                # (Vector{Integer}(1:n_insts), 1:n_insts)
                balanced_dataset_slice(
                    # dataset,
                    Y,
                    [dataseed];
                    ninstances_per_class=floor(Int, minimum(class_count) * 1.0),
                    also_return_discarted=false
                )[1]
            else
                balanced_dataset_slice(
                    # dataset,
                    Y,
                    [dataseed];
                    ninstances_per_class=floor(Int, minimum(class_count) * split_threshold),
                    also_return_discarted=true
                )[1]
            end
        end) for dataseed in todo_dataseeds])
    end

    println("Dataseeds = $(todo_dataseeds)")

    # for (dataseed,data_slice) in dataset_slices
    #   println("class_distribution: ")
    #   println(StatsBase.countmap(dataset[2][data_slice]))
    #   println("...")
    #   break # Note: Assuming this print is the same for all dataseeds
    # end
    # println()

    if dry_run == :dataset_only
        continue
    end

    ##############################################################################
    ##############################################################################
    ##############################################################################

    if dry_run == false
        exec_scan(
            params_namedtuple,
            dataset;
            issurelymultimodal=true,
            ### Training params
            train_seed=train_seed,
            modal_args=cur_modal_args,
            tree_args=tree_args,
            forest_args=forest_args,
            n_forest_runs=n_forest_runs,
            optimize_forest_computation=optimize_forest_computation,
            nsdt_args=nsdt_args,
            n_nsdt_folds=n_nsdt_folds,
            ### Dataset params
            # split_threshold                 =   split_threshold,
            data_modal_args=cur_data_modal_args,
            dataset_slices=dataset_slices,
            round_dataset_to_datatype=round_dataset_to_datatype,
            use_training_form=use_training_form,
            ### Run params
            results_dir=results_dir,
            data_savedir=data_savedir,
            model_savedir=model_savedir,
            # logger                          =   logger,
            timing_mode=timing_mode,
            ### Misc
            save_datasets=save_datasets,
            skip_training=skip_training,
            callback=(dataseed) -> begin
                # Add this step to the "history" of already computed iteration
                push_iteration_to_history!(history, (params_namedtuple, dataseed))
                save_history(iteration_progress_json_file_path, history)
            end
        )
    end
end

println("Done!")
println("# Iterations $(n_interations_done)/$(n_interations)")
println("Complete..Exit.")

# Notify the Telegram Bot
@error "Done!"

# close(logfile_io);

# !isinteractive() && exit(0)
