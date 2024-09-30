using CSV, DataFrames, JLD2
using Audio911
using ModalDecisionTrees
using SoleDecisionTreeInterface
using MLJ, Random
using StatsBase, Catch22
using CategoricalArrays
# using Plots

include("../utils.jl")
include("../audio_utils.jl")
include("../modal.jl")

# -------------------------------------------------------------------------- #
#                             global parameters                              #
# -------------------------------------------------------------------------- #
featset = (:mfcc, :f0, :spectrals)

audioparams = let sr = 8000
    (
        sr = sr,
        norm = true,
        speech_detect = true,
        sdetect_thresholds=(0,0), 
        sdetect_spread_threshold=0.02,
        nfft = 256,
        mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
        mel_nbands = 26,
        mfcc_ncoeffs = 13,
        mel_freqrange = (0, round(Int, sr / 2)),
    )
end

# min_length = 11500
# min_samples = 400

# only for debugging
min_length = 16000
min_samples = 38

features = :catch9
# features = :minmax
# features = :custom

# modal analysis
nwindows = 20
relative_overlap = 0.05

# partitioning
train_ratio = 0.8
train_seed = 11
rng = Random.MersenneTwister(train_seed)

# -------------------------------------------------------------------------- #
#                       experiment specific parameters                       #
# -------------------------------------------------------------------------- #
classes = :emo2bins
# classes = :emo3bins
# classes = :emo8bins

if classes == :emo2bins
    classes_dict = Dict{String,String}(
        "01" => "positive",
        "02" => "positive",
        "03" => "positive",
        "04" => "negative",
        "05" => "negative",
        "06" => "negative",
        "07" => "negative",
        "08" => "positive"
    )
elseif classes == :emo3bins
    classes_dict = Dict{String,String}(
        "01" => "neutral",
        "02" => "neutral",
        "03" => "positive",
        "04" => "negative",
        "05" => "negative",
        "06" => "negative",
        "07" => "negative",
        "08" => "positive"
    )    
elseif classes == :emo8bins
    classes_dict = Dict{String,String}(
        "01" => "neutral",
        "02" => "calm",
        "03" => "happy",
        "04" => "sad",
        "05" => "angry",
        "06" => "fearful",
        "07" => "disgust",
        "08" => "surprised"
    )
end

classes_func(row) = match(r"^(?:[^-]*-){2}([^-]*)", row.filename)[1]

wav_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"

# source_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"
# dest_path = "/home/paso/Documents/Aclai/ItaData2024/jld2_files/emotion"
# wav_jld2_name = "ravdess_wavfiles"

# -------------------------------------------------------------------------- #
#                                   main                                     #
# -------------------------------------------------------------------------- #
df = collect_audio_from_folder(wav_path; audioparams=audioparams)
labels = read_filenames(df, classes_dict, classes_func)
merge_df_labels!(df, labels)
sort_df!(df, :length; rev=true)
df = trimlength_df(df, :label, :length, :audio; min_length=min_length, min_samples=min_samples, sr=audioparams.sr)
X, y, variable_names = afe(df, featset, audioparams)
# propositional_analisys(X, y, variable_names, classes)
modal_sole_dt = modal_analisys(X, y; variable_names=variable_names, features=features, nwindows=nwindows, relative_overlap=relative_overlap, train_ratio=train_ratio, rng=rng)

@info("Done.")