using Pkg
# Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.activate("/home/paso/Documents/Aclai/audio-rules2024")

using CSV, DataFrames, JLD2
using Audio911
using Catch22
using StatsBase
using Plots

include("../utils.jl")

# -------------------------------------------------------------------------- #
#                                 parameters                                 #
# -------------------------------------------------------------------------- #
sr = 8000
audioparams = (
    sr = sr,
    nfft = 256,
    mel_scale = :mel_htk, # :mel_htk, :mel_slaney, :erb, :bark, :semitones, :tuned_semitones
    mel_nbands = 26,
    mfcc_ncoeffs = 13,
    mel_freqrange = (50, round(Int, sr / 2)),
    featset = (:mfcc, :deltas)
)

labels = [:file_cough, :label]
ids = [:filename, :file_cough]

min_length = 16000
min_samples = 400

csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/sounddr_data_passwordFPT_Software/data.csv"
dest_path = "/home/paso/Documents/Aclai/Experiments/sounddr/jld2_files"

wav_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/sounddr_data_passwordFPT_Software/breathe_mouth"
# wav_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/sounddr_data_passwordFPT_Software/breathe_nose"
# wav_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/databases/sounddr_data_passwordFPT_Software/cough"

audiotype = match(r"/([^/]+)$", wav_path).captures[1]
fname = string("sounddr_", audiotype, "_$(audioparams.mel_scale)_$(audioparams.mel_nbands)")

# -------------------------------------------------------------------------- #
#                                    main                                    #
# -------------------------------------------------------------------------- #
df = collect_audio_from_folder(wav_path, sr=audioparams.sr)
csvdf = csv2df(csv_path; labels=labels, header=true)
merge_df_csv!(df, csvdf; id_df=ids[1], id_csv=ids[2])

get_audio_length!(df, :audio, :audio_length)
trm_df = trimlength_df(df, :label, :audio_length, :audio; min_length=min_length, min_samples=min_samples)
afe_df = audio_features(trm_df; audioparams=audioparams)

y = trm_df[:, :label]
X = DataFrame()

X[!, :label] = String[]
for i in 1:size(afe_df[1], 2)
    colname = "a$i"
    X[!, colname] = Vector{Float64}[]
end
for i in axes(y, 1)
    push!(X, vcat(string(y[i]), vcat.(k for k in eachcol(afe_df[i]))))
end
save_jld2(X, string(dest_path, "/", fname, ".jld2"); labels=size(y, 2))
