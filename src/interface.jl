# -------------------------------------------------------------------------- #
#                           get interesting rules                            #
# -------------------------------------------------------------------------- #
function get_interesting_rules(
    
)
    df = collect_audio_from_folder(wav_path; audioparams=audioparams)
    labels = collect_classes(df, classes_dict, classes_func)
    merge_df_labels!(df, labels)
    sort_df!(df, :length; rev=true)
    df = trimlength_df(df, :label, :length, :audio; min_length=min_length, min_samples=min_samples, sr=audioparams.sr)
    X, y, variable_names = afe(df, featset, audioparams)
    prop_sole_dt = propositional_analisys(X, y, variable_names=variable_names, features=features, train_ratio=train_ratio, rng=rng)
    modal_sole_dt = modal_analisys(X, y; variable_names=variable_names, features=features, nwindows=nwindows, relative_overlap=relative_overlap, train_ratio=train_ratio, rng=rng)
    interesting_rules(prop_sole_dt, modal_sole_dt; variable_names=variable_names)
end