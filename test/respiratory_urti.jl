using DataFrames, JLD2
using SoleAudio, Random
# using Plots

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
classes = :URTI
# classes = :Bronchiectasis
# classes = :Bronchiolitis
# classes = :resp4bins

jld2_file = string("respiratory_", classes)

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

# audioparams = let sr = 8000
#     (
#         sr = sr,
#         norm = true,
#         speech_detect = false,
#         nfft = 256,
#         mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
#         mel_nbands = 32,
#         mfcc_ncoeffs = 16,
#         mel_freqrange = (300, round(Int, sr / 2)),
#     )
# end

audioparams = let sr = 8000
    (
        sr = sr,
        norm = true,
        speech_detect = false,
        nfft = 256,
        mel_scale = :bark, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
        mel_nbands = 26,
        mfcc_ncoeffs = 13,
        mel_freqrange = (300, round(Int, sr / 2)),
    )
end

min_length = 120000
min_samples = 14

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
    csv_file=csv_file,
    classes_dict=classes_dict,
    fragmented=fragmented,
    frag_func=frag_func,
    header=header,
    id_labels=id_labels,
    label_labels=label_labels,
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