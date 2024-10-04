using DataFrames, JLD2
using SoleAudio, Random
# using Plots

# TODO
# scrivi un file text con tutti i settaggi usati
# output formattato per latex su un file.tex

# -------------------------------------------------------------------------- #
#                       experiment specific parameters                       #
# -------------------------------------------------------------------------- #
# wav_path ="/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database/audio_partitioned"
# csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database"
wav_path ="/home/paso/datasets/health_recognition/Respiratory_Sound_Database/audio_partitioned"
csv_path = "/home/paso/datasets/health_recognition/Respiratory_Sound_Database"

csv_file = csv_path * "/" * "patient_diagnosis.csv"

# classes = :Pneumonia
# classes = :COPD
# classes = :URTI
# classes = :Bronchiectasis
classes = :Bronchiolitis
# classes = :resp4bins

if classes == :Pneumonia
    classes_dict = Dict{String,String}(
        "Pneumonia" => "sick",
        "Healthy" => "healthy",
    )
elseif classes == :COPD
    classes_dict = Dict{String,String}(
        "COPD" => "sick",
        "Healthy" => "healthy",
    )
elseif classes == :URTI
    classes_dict = Dict{String,String}(
        "URTI" => "sick",
        "Healthy" => "healthy",
    )
elseif classes == :Bronchiectasis
    classes_dict = Dict{String,String}(
        "Bronchiectasis" => "sick",
        "Healthy" => "healthy",
    )
elseif classes == :Bronchiolitis
    classes_dict = Dict{String,String}(
        "Bronchiolitis" => "sick",
        "Healthy" => "healthy",
    )
elseif classes == :resp4bins
    classes_dict = Dict{String,String}(
        "Pneumonia" => "pneumonia",
        "COPD" => "copd",
        "URTI" => "urti",
        "Healthy" => "healthy",
    )
end

fragmented = true
frag_func(filename) = match(r"^(\d+)", filename)[1]

header = false
id_labels = :Column1
label_labels = :Column2

# -------------------------------------------------------------------------- #
#                             global parameters                              #
# -------------------------------------------------------------------------- #
featset = (:mel, :mfcc, :spectrals)

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
    csv_file=csv_file,
    classes_dict=classes_dict,
    fragmented=fragmented,
    frag_func=frag_func,
    header=header,
    id_labels=id_labels,
    label_labels=label_labels,
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

jldsave("respiratory_bronchiolitis.jld2", true; irules)