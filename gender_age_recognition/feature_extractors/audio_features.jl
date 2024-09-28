function _audio_features_extractor(
    x::Vector{Float64},
    sr::Int64,
    profile::Symbol,
    ds_type::Symbol,
    stft_length::Int64,
    freq_range::Tuple{Int64, Int64},
    mel_bands::Int64,
    mfcc_coeffs::Int64,
)
    if profile == :full
        # versione ufficiale
        setup = AudioSetup(
            sr=sr,
            # fft
            window_type=(:hann, :periodic),
            window_length=stft_length,
            overlap_length=Int(round(stft_length * 0.500)),
            window_norm=false,
            # spectrum
            freq_range=freq_range,
            spectrum_type=:power, # :power, :magnitude
            # mel
            mel_style=:htk, # :htk, :slaney
            mel_bands=mel_bands,
            filterbank_design_domain=:linear,
            filterbank_normalization=:bandwidth, # :bandwidth, :area, :none
            frequency_scale=:mel,
            # mfcc
            mfcc_coeffs=mfcc_coeffs,
            normalization_type=:dithered, # :standard, :dithered
            rectification=:log,
            log_energy_source=:standard, # :standard (after windowing), :mfcc
            log_energy_pos=:none, #:append, :replace, :none
            delta_window_length=9,
            delta_matrix=:transposed, # :standard, :transposed
            # spectral
            spectral_spectrum=:linear, # :linear, :mel
        )

        data = AudioData(
            x=x,
        )

        get_stft!(data, setup)
        mel_spectrogram(data, setup)
        _mfcc(data, setup)
        f0(data, setup)
        # setup.stft.freq_range = (0, 1000) # verifica che 1000 < sr/2, perfetto per gender, rovina age
        if ds_type == :gender && sr / 2 > 2000
            setup.stft.freq_range = (0, 2000)
        end
        lin_spectrogram(data, setup)
        spectral_features(data, setup)

        return vcat(
            (
                data.mel_spectrogram',
                data.mfcc_coeffs',
                data.mfcc_delta',
                data.mfcc_deltadelta',
                data.spectral_centroid',
                data.spectral_crest',
                data.spectral_decrease',
                data.spectral_entropy',
                data.spectral_flatness',
                data.spectral_flux',
                data.spectral_kurtosis',
                data.spectral_rolloff',
                data.spectral_skewness',
                data.spectral_slope',
                data.spectral_spread',
                data.f0',
            )...,
        )

        # # versione matlab certificata identica
        # setup = AudioSetup(
        #     sr=sr,
        #     # fft
        #     window_type=(:hann, :periodic),
        #     window_length=stft_length,
        #     overlap_length=Int(round(stft_length * 0.500)),
        #     window_norm=true,
        #     # spectrum
        #     freq_range=freq_range,
        #     spectrum_type=:power, # :power, :magnitude
        #     # mel
        #     mel_style=:htk, # :htk, :slaney
        #     mel_bands=mel_bands,
        #     filterbank_design_domain=:linear,
        #     filterbank_normalization=:bandwidth, # :bandwidth, :area, :none
        #     frequency_scale=:mel,
        #     # mfcc
        #     mfcc_coeffs=mfcc_coeffs,
        #     normalization_type=:standard, # :standard, :dithered
        #     rectification=:log,
        #     log_energy_source=:standard, # :standard (after windowing), :mfcc
        #     log_energy_pos=:none, #:append, :replace, :none
        #     delta_window_length=9,
        #     delta_matrix=:standard, # :standard, :transposed
        #     # spectral
        #     spectral_spectrum=:linear # :linear, :mel
        # )

        # data = AudioData(
        #     x=x
        # )

        # get_stft!(data, setup)
        # mel_spectrogram(data, setup)
        # _mfcc(data, setup)
        # lin_spectrogram(data, setup)
        # spectral_features(data, setup)
        # f0(data, setup)

        # return vcat((
        #     data.mel_spectrogram',
        #     data.mfcc_coeffs',
        #     data.mfcc_delta',
        #     data.mfcc_deltadelta',
        #     data.spectral_centroid',
        #     data.spectral_crest',
        #     data.spectral_decrease',
        #     data.spectral_entropy',
        #     data.spectral_flatness',
        #     data.spectral_flux',
        #     data.spectral_kurtosis',
        #     data.spectral_rolloff',
        #     data.spectral_skewness',
        #     data.spectral_slope',
        #     data.spectral_spread',
        #     data.f0'
        # )...)

        # # versione audioflux identica, tranne f0
        # setup = AudioSetup(
        # 	sr = sr,
        # 	# fft
        # 	window_type =(:hann, :periodic),
        # 	window_length = stft_length,
        # 	overlap_length = Int(round(stft_length * 0.500)),
        # 	window_norm = false,
        # 	# spectrum
        # 	freq_range = freq_range,
        # 	spectrum_type = :power, # :power, :magnitude
        # 	# mel
        # 	mel_style = :htk, # :htk, :slaney
        # 	mel_bands = mel_bands,
        # 	filterbank_design_domain = :linear,
        # 	filterbank_normalization = :bandwidth, # :bandwidth, :area, :none
        # 	frequency_scale = :mel,
        # 	# mfcc
        # 	mfcc_coeffs = mfcc_coeffs,
        # 	normalization_type = :dithered, # :standard, :dithered
        # 	rectification = :log,
        # 	log_energy_source = :standard, # :standard (after windowing), :mfcc
        # 	log_energy_pos = :none, #:append, :replace, :none
        # 	delta_window_length = 9,
        # 	delta_matrix = :transposed, # :standard, :transposed
        # 	# spectral
        # 	spectral_spectrum = :linear, # :linear, :mel
        # )

        # data = AudioData(
        # 	x = x,
        # )

        # get_stft!(data, setup)
        # mel_spectrogram(data, setup)
        # _mfcc(data, setup)
        # setup.spectrum_type = :magnitude
        # get_stft!(data, setup)
        # lin_spectrogram(data, setup)
        # spectral_features(data, setup)
        # # f0(data, setup)

        # return vcat(
        # 	(
        # 		data.mel_spectrogram',
        # 		data.mfcc_coeffs',
        # 		data.mfcc_delta',
        # 		data.mfcc_deltadelta',
        # 		# data.spectral_centroid',
        # 		# data.spectral_crest',
        # 		# data.spectral_decrease',
        # 		# data.spectral_entropy',
        # 		# data.spectral_flatness',
        # 		# data.spectral_flux',
        # 		# data.spectral_kurtosis',
        # 		# data.spectral_rolloff',
        # 		# data.spectral_skewness',
        # 		# data.spectral_slope',
        # 		# data.spectral_spread',
        # 		# data.f0',
        # 	)...,
        # )

    elseif profile == :gender # le delta peggiorano, visto, ma serve l'ufficialità
        setup = AudioSetup(
            sr=sr,
            # fft
            window_type=(:hann, :periodic),
            window_length=stft_length,
            overlap_length=Int(round(stft_length * 0.500)),
            window_norm=true,
            # spectrum
            freq_range=freq_range,
            spectrum_type=:power, # :power, :magnitude
            # mel
            mel_style=:htk, # :htk, :slaney
            mel_bands=mel_bands,
            filterbank_design_domain=:linear,
            filterbank_normalization=:bandwidth, # :bandwidth, :area, :none
            frequency_scale=:mel,
            # mfcc
            mfcc_coeffs=mfcc_coeffs,
            normalization_type=:dithered, # :standard, :dithered
            rectification=:log,
            log_energy_source=:standard, # :standard (after windowing), :mfcc
            log_energy_pos=:none, #:append, :replace, :none
            delta_window_length=9,
            delta_matrix=:standard, # :standard, :transposed
            # spectral
            spectral_spectrum=:linear, # :linear, :linear_focused, :mel
        )

        data = AudioData(
            x=x,
        )

        get_stft!(data, setup)
        mel_spectrogram(data, setup)
        _mfcc(data, setup)
        f0(data, setup)
        setup.stft.freq_range = (0, 1000) # verifica che 1000 < sr/2
        lin_spectrogram(data, setup)
        spectral_features(data, setup)

        return vcat(
            (
                data.mel_spectrogram',
                data.mfcc_coeffs',
                # data.mfcc_delta',
                # data.mfcc_deltadelta',
                data.spectral_centroid',
                data.spectral_crest',
                data.spectral_decrease',
                data.spectral_entropy',
                data.spectral_flatness',
                data.spectral_flux',
                data.spectral_kurtosis',
                data.spectral_rolloff',
                data.spectral_skewness',
                data.spectral_slope',
                data.spectral_spread',
                data.f0',
            )...,
        )

    elseif profile == :age
        setup = AudioSetup(
            sr=sr,
            # fft
            window_type=(:hann, :periodic),
            window_length=stft_length,
            overlap_length=Int(round(stft_length * 0.500)),

            # window_norm = true,
            window_norm=false,
            # spectrum
            freq_range=freq_range,
            spectrum_type=:power, # :power, :magnitude
            # mel
            mel_style=:htk, # :htk, :slaney
            mel_bands=mel_bands,
            filterbank_design_domain=:linear,
            filterbank_normalization=:bandwidth, # :bandwidth, :area, :none
            frequency_scale=:mel,
            # mfcc
            mfcc_coeffs=mfcc_coeffs,
            normalization_type=:dithered, # :standard, :dithered
            rectification=:log,
            log_energy_source=:standard, # :standard (after windowing), :mfcc
            log_energy_pos=:none, #:append, :replace, :none
            delta_window_length=9,
            delta_matrix=:standard, # :standard, :transposed
            # spectral
            spectral_spectrum=:linear, # :linear, :linear_focused, :mel
        )

        data = AudioData(
            x=x,
        )

        get_stft!(data, setup)
        mel_spectrogram(data, setup)
        _mfcc(data, setup)
        f0(data, setup)

        # setup.stft.freq_range = (0, 1000) # verifica che 1000 < sr/2
        setup.stft.freq_range = (0, 2000)

        lin_spectrogram(data, setup)
        spectral_features(data, setup)

        return vcat(
            (
                # data.mel_spectrogram',
                data.mfcc_coeffs',
                # data.mfcc_delta',
                # data.mfcc_deltadelta',
                data.spectral_centroid',
                data.spectral_crest',
                data.spectral_decrease',
                # data.spectral_entropy',
                data.spectral_flatness',
                data.spectral_flux',
                # data.spectral_kurtosis',
                # data.spectral_rolloff',
                # data.spectral_skewness',
                # data.spectral_slope',
                # data.spectral_spread',
                data.f0',
            )...,
        )

    elseif profile == :wavelets
        window_length = 256
        freq_range = (80, 4000)

        setup = AudioSetup(
            sr=sr,
            # fft
            window_type=(:hann, :periodic),
            window_length=window_length,
            overlap_length=round(Int, window_length * 0.500),
            window_norm=false,
            # spectrum
            freq_range=freq_range,
            spectrum_type=:magnitude, # :power, :magnitude
            # mel
            mel_style=:htk, # :htk, :slaney
            # mel_bands=mel_bands,
            filterbank_design_domain=:linear,
            filterbank_normalization=:bandwidth, # :bandwidth, :area, :none
            frequency_scale=:mel,
            # mfcc
            # mfcc_coeffs=mfcc_coeffs,
            normalization_type=:dithered, # :standard, :dithered
            rectification=:log,
            log_energy_source=:standard, # :standard (after windowing), :mfcc
            log_energy_pos=:none, #:append, :replace, :none
            delta_window_length=9,
            delta_matrix=:transposed, # :standard, :transposed
            # spectral
            spectral_spectrum=:mel, # :linear, :linear_focused, :mel
        )

        data = AudioData(
            x=x,
        )

        # println("parte")
        cwt_spectrum, _ = cwt(data.x, setup.sr, freq_range=(80, 4000))
        # println("cwt calcolata")
        setup.spectrum_type == :power ? cwt_spectrum = real(cwt_spectrum' .* conj(cwt_spectrum')) : cwt_spectrum = abs.(cwt_spectrum')
        data.mel_spectrogram = cwt_windowing(cwt_spectrum, 256)
        setup.mel_bands = size(data.mel_spectrogram, 2)
        setup.mfcc_coeffs = floor(Int, size(data.mel_spectrogram, 2) / 2)

        _mfcc(data, setup)
        # f0(data, setup)
        # spectral_features(data, setup)

        return vcat(
            (
                data.mel_spectrogram',
                data.mfcc_coeffs',
                # data.mfcc_delta',
                # data.mfcc_deltadelta',
                # data.spectral_centroid',
                # data.spectral_crest',
                # data.spectral_decrease',
                # data.spectral_entropy',
                # data.spectral_flatness',
                # data.spectral_flux',
                # data.spectral_kurtosis',
                # data.spectral_rolloff',
                # data.spectral_skewness',
                # data.spectral_slope',
                # data.spectral_spread',
                # data.f0',
            )...,
        )

    else
        error("Unknown profile type: $profile.")
    end

end

# log_energy=:none, deciso così perchè i risultati hanno una deviazione standard molto bassa < 1.
# quindi sembra essere più stabile.

# analizzato quanto sspeech trova informative le delta: la versione :transposed è poco informativa.
# non vengono prese in considerazione da sspeech, questo potrebbe essere il motivo per cui audioflux è più performate di matlab.

# verificato che il setup base di matlab (mel_bands=32, windowing .3 overlap .2) è peggiorativo.
# window_length=Int(round(0.03 * sr)),
# overlap_length=Int(round(0.02 * sr)),

# i NaN vengono generati da pochissimi file, 1 su 1000. quindi si preferisce visualizzare i file incriminato e cancellarlo

# provato nelle spectral sia :power che :magnitude, ma le differenze sono ininfluenti
# restringendo il campo delle spectral entro i 1000hz si ottengono i risultati buoni di audioflux sbagliata