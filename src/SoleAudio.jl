module SoleAudio

using CSV, DataFrames, JLD2
using Audio911
using ModalDecisionTrees
using SoleDecisionTreeInterface
using MLJ, Random
using StatsBase, Catch22
using CategoricalArrays
using Printf
# using Plots

include("analysis.jl")
include("audio_utils.jl")
include("utils.jl")
include("rules.jl")
include("interface.jl")

export get_df_from_rawaudio, get_interesting_rules

# debug
export collect_audio_from_folder, collect_classes, merge_df_labels!
export sort_df!, trimlength_df, vnames_builder

end