using DataFrames, JLD2
using SoleAudio, Random
# using Plots

# -------------------------------------------------------------------------- #
#                       experiment specific parameters                       #
# -------------------------------------------------------------------------- #
# wav_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"
wav_path = "/home/paso/datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"

# classes = :emo2bins
# classes = :emo3bins
classes = :emo8bins

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
        "05" => "negative",
        "07" => "negative",
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

jld2_file = string("ravdess_", classes)

# classes will be taken from audio filename, no csv available
classes_func(row) = match(r"^(?:[^-]*-){2}([^-]*)", row.filename)[1]

# -------------------------------------------------------------------------- #
#                             global parameters                              #
# -------------------------------------------------------------------------- #
featset = (:mel, :mfcc, :f0, :spectrals)

# audioparams = let sr = 8000
#     (
#         sr = sr,
#         norm = true,
#         speech_detect = true,
#         sdetect_thresholds=(0,0), 
#         sdetect_spread_threshold=0.02,
#         nfft = 256,
#         mel_scale = :semitones, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
#         mel_nbands = 26,
#         mfcc_ncoeffs = 13,
#         mel_freqrange = (100, round(Int, sr / 2)),
#     )
# end

audioparams = let sr = 8000
    (
        sr = sr,
        norm = true,
        speech_detect = true,
        sdetect_thresholds=(0,0), 
        sdetect_spread_threshold=0.02,
        nfft = 256,
        mel_scale = :erb, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
        mel_nbands = 26,
        mfcc_ncoeffs = 13,
        mel_freqrange = (100, round(Int, sr / 2)),
    )
end

min_length = 11500
min_samples = 50

features = :catch9
# features = :minmax
# features = :custom

# modal analysis
nwindows = 20
relative_overlap = 0.05

# partitioning
# train_ratio = 0.8
# train_seed = 1
train_ratio = 0.7
train_seed = 9
rng = Random.MersenneTwister(train_seed)
Random.seed!(train_seed)

# -------------------------------------------------------------------------- #
#                                   main                                     #
# -------------------------------------------------------------------------- #
df = get_df_from_rawaudio(
    wav_path=wav_path,
    classes_dict=classes_dict,
    classes_func=classes_func,
    audioparams=audioparams,
)

irules = get_interesting_rules(
    df;
    featset=featset,
    audioparams=audioparams,
    min_length=min_length,
    min_samples=min_samples,
    features=features,
    nwindows=nwindows,
    relative_overlap=relative_overlap,
    train_ratio=train_ratio,
    rng=rng,
)

println(irules)

jldsave(jld2_file * ".jld2", true; irules)
@info "Done."
