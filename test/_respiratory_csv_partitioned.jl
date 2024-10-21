using DataFrames, JLD2, CSV
using Audio911
# using Plots

# wav_path ="/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database/audio_partitioned"
# csv_path = "/home/paso/Documents/Aclai/Datasets/health_recognition/Respiratory_Sound_Database"
wav_path ="/home/paso/datasets/health_recognition/Respiratory_Sound_Database/audio_partitioned"
csv_path = "/home/paso/datasets/health_recognition/Respiratory_Sound_Database"
save_csv_path = "/home/paso/datasets/health_recognition/Respiratory_Sound_Database"
csv_name = "patient_diagnosis_partitioned"

csv_file = csv_path * "/" * "patient_diagnosis.csv"

# frag_func(filename) = match(r"^(\d+)", filename)[1]
frag_func(filename) = match(r"^(\d+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+)", filename)[1]

audioparams = let sr = 16000
    (
        sr = sr,
        norm = true,
        speech_detect = false,
    )
end

csvdf = CSV.read(csv_file, DataFrame, header=false)
column_dict = Dict(csvdf.Column1 .=> csvdf.Column2)

params(x::Dict, audioparams::NamedTuple) = (x[k] => getfield(audioparams, Symbol(k)) for k in keys(x) if haskey(audioparams, Symbol(k)))

function walk_audio_dir!(df::DataFrame, path::String; audioparams::NamedTuple)
    norm = get(audioparams, :norm, false)
    speech_detect = get(audioparams, :speech_detect, false)
    sd_params = Dict(:sd_thr => :thresholds, :sd_spread_thr => :spread_threshold)

    for (root, _, files) in walkdir(path)
        for file in filter(f -> any(occursin.([".wav", ".flac", ".mp3"], f)), files)
            audio = load_audio(joinpath(root, file), audioparams.sr; norm=norm)
            speech_detect && begin audio, _ = speech_detector(audio; params(sd_params, audioparams)...) end
            push!(df, hcat(split(file, ".")[1], size(audio.data, 1), [audio.data]))
        end
    end
end

csv_df = DataFrame(filename=String[], diagnosis=String[])
@info "Defragmenting audio files..."
dff = DataFrame(filename=String[], length=Int64[], audio=AbstractArray{<:AbstractFloat}[])
walk_audio_dir!(dff, wav_path; audioparams=audioparams)


for i in eachrow(dff)
    push!(csv_df, hcat(i.filename, column_dict[parse(Int, split(frag_func(i.filename), "_")[1])]))
end

CSV.write(string(save_csv_path, "/", csv_name, ".csv"), csv_df)
