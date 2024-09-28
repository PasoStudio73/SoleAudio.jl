using PyCall

lib = pyimport("librosa")

function librosa_extractor(
    x::Vector{Float64},
    sr::Int64,
    profile::Symbol,
    stft_length::Int64,
    freq_range::Tuple{Int64, Int64},
    mel_bands::Int64,
    mfcc_coeffs::Int64,
    ext::Symbol
)
    if profile == :full
        # compute mel spectrogram
        mel_spec = lib.feature.melspectrogram(
            y=x,
            sr=sr,
            n_fft=stft_length,
            hop_length=Int(round(stft_length * 0.500)),
            win_length=stft_length,
            # window=scipy.signal.windows.hann,
            power=2, # 1: magnitude, 2: power
            n_mels=mel_bands,
            htk=true
        )

        # compute mfcc and deltas
        mfcc_coeffs = lib.feature.mfcc(
            y=x,
            sr=sr,
            n_mfcc=mfcc_coeffs,
            dct_type=2,
            norm="ortho",
            lifter=0,
            n_fft=stft_length,
            hop_length=Int(round(stft_length * 0.500)),
            win_length=stft_length,
            # window=scipy.signal.windows.hann,
            power=2,
            n_mels=mel_bands,
            htk=true,
        )
        mfcc_delta = lib.feature.delta(mfcc_coeffs)
        mfcc_deltadelta = lib.feature.delta(mfcc_coeffs, order=2)

        # compute spectral features
        centroid = lib.feature.spectral_centroid(
            y=x,
            sr=sr,
            n_fft=stft_length,
            hop_length=Int(round(stft_length * 0.500)),
            win_length=stft_length
        )

        flatness = lib.feature.spectral_flatness(
            y=x,
            n_fft=stft_length,
            hop_length=Int(round(stft_length * 0.500)),
            win_length=stft_length
        )

        rolloff = lib.feature.spectral_rolloff(
            y=x,
            sr=sr,
            n_fft=stft_length,
            hop_length=Int(round(stft_length * 0.500)),
            win_length=stft_length
        )

        ### features extended
        # v1: audio911
        if ext == :a911 # spectral features matlab like version
            setup = AudioSetup(
                sr=sr,
                # fft
                window_type=(:hann, :periodic),
                window_length=stft_length,
                overlap_length=Int(round(stft_length * 0.500)),
                window_norm=false,
                # spectrum
                freq_range=(0, Int(sr/2)), # come lavora matlab, non la nostra
                spectrum_type=:power, # :power, :magnitude
                # spectral
                spectral_spectrum=:linear # :linear, :mel
            )

            data = AudioData(
                x=x
            )

            get_stft!(data, setup)
            lin_spectrogram(data, setup)
            spectral_features(data, setup)
            f0(data, setup) # pay attention to fft length

            return vcat((
                mel_spec[:, 2:end-1],
                mfcc_coeffs[:, 2:end-1],
                mfcc_delta[:, 2:end-1],
                mfcc_deltadelta[:, 2:end-1],
                centroid[:, 2:end-1],
                data.spectral_crest',
                data.spectral_decrease',
                data.spectral_entropy',
                flatness[:, 2:end-1],
                data.spectral_flux',
                data.spectral_kurtosis',
                rolloff[:, 2:end-1],
                data.spectral_skewness',
                data.spectral_slope',
                data.spectral_spread',
                data.f0'
            )...)

            # v2 audioflux
        elseif ext == :af
            # compute spectral features
            s_obj = af.BFT(
                num=Int64(stft_length / 2 + 1),
                radix2_exp=Int64(log2(stft_length)),
                samplate=sr,
                high_fre=sr / 2,
                window_type=af.type.WindowType.HANN,
                slide_length=round(Integer, stft_length * 0.500),
                scale_type=af.type.SpectralFilterBankScaleType.LINEAR,
                data_type=af.type.SpectralDataType.MAG,
            )
            s_spec_arr = s_obj.bft(x)
            s_spec_arr = abs.(s_spec_arr)

            s_spectral_obj = af.Spectral(
                num=s_obj.num,
                fre_band_arr=s_obj.get_fre_band_arr()
            )
            s_n_time = length(s_spec_arr[2, :])
            s_spectral_obj.set_time_length(s_n_time)

            # centroid_arr = s_spectral_obj.centroid(s_spec_arr)
            crest_arr = s_spectral_obj.crest(s_spec_arr)
            decrease_arr = s_spectral_obj.decrease(s_spec_arr)
            entropy_arr = s_spectral_obj.entropy(s_spec_arr)
            # flatness_arr = s_spectral_obj.flatness(s_spec_arr)
            flux_arr = s_spectral_obj.flux(s_spec_arr)
            kurtosis_arr = s_spectral_obj.kurtosis(s_spec_arr)
            # rolloff_arr = s_spectral_obj.rolloff(s_spec_arr)
            skewness_arr = s_spectral_obj.skewness(s_spec_arr)
            slope_arr = s_spectral_obj.slope(s_spec_arr)
            spread_arr = s_spectral_obj.spread(s_spec_arr)

            # estimate pitch
            pitch_obj = af.PitchYIN(
                samplate=sr,
                radix2_exp=Int64(log2(stft_length)),
                slide_length=round(Integer, stft_length * 0.500),
            )

            fre_arr, _, _ = pitch_obj.pitch(x)

            return vcat((
                mel_spec[:, 2:end-1],
                mfcc_coeffs[:, 2:end-1],
                mfcc_delta[:, 2:end-1],
                mfcc_deltadelta[:, 2:end-1],
                centroid[:, 2:end-1],
                crest_arr',
                decrease_arr',
                entropy_arr',
                flatness[:, 2:end-1],
                flux_arr',
                kurtosis_arr',
                rolloff[:, 2:end-1],
                skewness_arr',
                slope_arr',
                spread_arr',
                fre_arr',
            )...)

        else
            error("Unknown librosa extension: $ext.")
        end

    else
        error("Unknown profile type: $profile.")
    end
end