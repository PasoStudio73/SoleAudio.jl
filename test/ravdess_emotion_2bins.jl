using DataFrames, JLD2
using SoleAudio, Random
# using Plots

# TODO
# scrivi un file text con tutti i settaggi usati
# output formattato per latex su un file.tex
# fra gli algoritmi di f0 includi il pagliarini

# -------------------------------------------------------------------------- #
#                       experiment specific parameters                       #
# -------------------------------------------------------------------------- #
wav_path = "/home/paso/Documents/Aclai/Datasets/emotion_recognition/Ravdess/audio_speech_actors_01-24"

# classes = :emo2bins
classes = :emo3bins
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

# classes will be taken from audio filename, no csv available
classes_func(row) = match(r"^(?:[^-]*-){2}([^-]*)", row.filename)[1]

# -------------------------------------------------------------------------- #
#                             global parameters                              #
# -------------------------------------------------------------------------- #
featset = (:mel, :mfcc, :f0, :spectrals)

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
min_length = 17000
min_samples = 6

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
#                                   main                                     #
# -------------------------------------------------------------------------- #
irules = get_interesting_rules(
    wav_path=wav_path,
    classes_dict=classes_dict,
    classes_func=classes_func,
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
