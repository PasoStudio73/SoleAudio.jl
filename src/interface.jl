# -------------------------------------------------------------------------- #
#                           get interesting rules                            #
# -------------------------------------------------------------------------- #
function get_df_from_rawaudio(;
    wav_path::String,
    csv_file::Union{String, Nothing}=nothing,
    # labels settings
    classes_dict::Dict{String, String},
    classes_func::Union{Function, Nothing}=nothing,
    fragmented::Bool=false,
    frag_func::Union{Function, Nothing}=nothing,
    header::Bool=false,
    # merge df labels settings
    sort_before_merge::Bool=true, 
    id_df::Symbol=:filename, 
    id_labels::Symbol=:filename,
    label_df::Symbol=:label,
    label_labels::Symbol=:label,
    # audio settings
    audioparams::NamedTuple=let sr=8000
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
    end,
)
    df = collect_audio_from_folder(wav_path; audioparams=audioparams, fragmented=fragmented, frag_func=frag_func)
    labels = isnothing(csv_file) ? 
        collect_classes(df, classes_dict; classes_func=classes_func) : 
        collect_classes(csv_file, classes_dict; id_labels=id_labels, label_labels=label_labels, header=header)
    merge_df_labels!(df, labels; sort_before_merge=sort_before_merge, id_df=id_df, id_labels=id_labels, label_df=label_df, label_labels=label_labels)
    return df
end

# -------------------------------------------------------------------------- #
#                           get interesting rules                            #
# -------------------------------------------------------------------------- #
function get_interesting_rules(
    df::DataFrame;
    # audio settings
    featset::Tuple=(:mel, :mfcc, :f0, :spectrals),
    audioparams::NamedTuple=let sr=8000
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
    end,
    # trim length settings
    splitlabel::Symbol = :label,
    lengthlabel::Symbol=:length,
    audiolabel::Symbol=:audio,
    min_length::Int64,
    min_samples::Int64,
    features::Symbol=:catch9,
    nwindows::Int64=20,
    relative_overlap::Float64=0.05,
    train_ratio::Float64=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    sort_df!(df, :length; rev=true)
    df = trimlength_df(df, splitlabel, lengthlabel, audiolabel; min_length=min_length, min_samples=min_samples, sr=audioparams.sr)
    X, y, variable_names = afe(df, featset, audioparams)
    prop_sole_dt = propositional_analisys(X, y, variable_names=variable_names, features=features, train_ratio=train_ratio, rng=rng)
    modal_sole_dt = modal_analisys(X, y; variable_names=variable_names, features=features, nwindows=nwindows, relative_overlap=relative_overlap, train_ratio=train_ratio, rng=rng)
    interesting_rules(prop_sole_dt, modal_sole_dt; features=features, variable_names=variable_names)
end