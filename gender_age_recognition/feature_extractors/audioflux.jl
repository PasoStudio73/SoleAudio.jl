using PyCall

af = pyimport("audioflux")

function audioflux_extractor(
	x::Vector{Float64},
	sr::Int64,
	profile::Symbol,
	stft_length::Int64,
	freq_range::Tuple{Int64, Int64},
	mel_bands::Int64,
	mfcc_coeffs::Int64,
)
	if profile == :full
		# compute mel spectrogram
		m_bft_obj = af.BFT(
			num = mel_bands,
			radix2_exp = Int64(log2(stft_length)),
			samplate = sr,
			low_fre = freq_range[1],
			high_fre = freq_range[2],
			window_type = af.type.WindowType.HANN,
			slide_length = round(Integer, stft_length * 0.500),
			scale_type = af.type.SpectralFilterBankScaleType.MEL,
			style_type = af.type.SpectralFilterBankStyleType.SLANEY,
			normal_type = af.type.SpectralFilterBankNormalType.BAND_WIDTH,
			data_type = af.type.SpectralDataType.POWER,
			is_reassign = false,
		)
		m_spec_arr = m_bft_obj.bft(x, result_type = 1)

		# compute mfcc and deltas
		m_xxcc_obj = af.XXCC(num = m_bft_obj.num)
		m_xxcc_obj.set_time_length(time_length = length(m_spec_arr[2, :]))
		m_spectral_obj = af.Spectral(
			num = m_bft_obj.num,
			fre_band_arr = m_bft_obj.get_fre_band_arr())
		m_n_time = length(m_spec_arr[2, :])
		m_spectral_obj.set_time_length(m_n_time)
		m_energy_arr = m_spectral_obj.energy(m_spec_arr)
		mfcc_arr, m_delta_arr, m_deltadelta_arr = m_xxcc_obj.xxcc_standard(
			m_spec_arr,
			m_energy_arr,
			cc_num = mfcc_coeffs,
			delta_window_length = 9,
			energy_type = af.type.CepstralEnergyType.IGNORE,
			rectify_type = af.type.CepstralRectifyType.LOG,
		)

		# compute spectral features
		s_obj = af.BFT(
			num = Int64(stft_length / 2 + 1),
			radix2_exp = Int64(log2(stft_length)),
			samplate = sr,
			high_fre = sr / 2,
			window_type = af.type.WindowType.HANN,
			slide_length = round(Integer, stft_length * 0.500),
			scale_type = af.type.SpectralFilterBankScaleType.LINEAR,
			data_type = af.type.SpectralDataType.MAG,
		)
		s_spec_arr = s_obj.bft(x)
		s_spec_arr = abs.(s_spec_arr)

		s_spectral_obj = af.Spectral(
			num = s_obj.num,
			fre_band_arr = s_obj.get_fre_band_arr(),
		)
		s_n_time = length(s_spec_arr[2, :])
		s_spectral_obj.set_time_length(s_n_time)

		centroid_arr = s_spectral_obj.centroid(s_spec_arr)
		crest_arr = s_spectral_obj.crest(s_spec_arr)
		decrease_arr = s_spectral_obj.decrease(s_spec_arr)
		entropy_arr = s_spectral_obj.entropy(s_spec_arr)
		flatness_arr = s_spectral_obj.flatness(s_spec_arr)
		flux_arr = s_spectral_obj.flux(s_spec_arr)
		kurtosis_arr = s_spectral_obj.kurtosis(s_spec_arr)
		rolloff_arr = s_spectral_obj.rolloff(s_spec_arr)
		skewness_arr = s_spectral_obj.skewness(s_spec_arr)
		slope_arr = s_spectral_obj.slope(s_spec_arr)
		spread_arr = s_spectral_obj.spread(s_spec_arr)

		# estimate pitch
		pitch_obj = af.PitchYIN(
			samplate = sr,
			radix2_exp = Int64(log2(stft_length)),
			slide_length = round(Integer, stft_length * 0.500),
		)

		fre_arr, _, _ = pitch_obj.pitch(x)

		return vcat((
			m_spec_arr,
			mfcc_arr,
			m_delta_arr,
			m_deltadelta_arr,
			centroid_arr',
			crest_arr',
			decrease_arr',
			entropy_arr',
			flatness_arr',
			flux_arr',
			kurtosis_arr',
			rolloff_arr',
			skewness_arr',
			slope_arr',
			spread_arr',
			fre_arr',
		)...)

	elseif profile == :gender
		# compute mel spectrogram
		m_bft_obj = af.BFT(
		    num=mel_bands,
		    radix2_exp=Int64(log2(stft_length)),
		    samplate=sr,
		    low_fre=freq_range[1],
		    high_fre=freq_range[2],
		    window_type=af.type.WindowType.HANN,
		    slide_length=round(Integer, stft_length * 0.500),
		    scale_type=af.type.SpectralFilterBankScaleType.MEL,
		    style_type=af.type.SpectralFilterBankStyleType.SLANEY,
		    normal_type=af.type.SpectralFilterBankNormalType.BAND_WIDTH,
		    data_type=af.type.SpectralDataType.POWER,
		    is_reassign=false
		)
		m_spec_arr = m_bft_obj.bft(x, result_type=1)

		# compute mfcc and deltas
		m_xxcc_obj = af.XXCC(num=m_bft_obj.num)
		m_xxcc_obj.set_time_length(time_length=length(m_spec_arr[2, :]))
		m_spectral_obj = af.Spectral(
		    num=m_bft_obj.num,
		    fre_band_arr=m_bft_obj.get_fre_band_arr())
		m_n_time = length(m_spec_arr[2, :])
		m_spectral_obj.set_time_length(m_n_time)
		m_energy_arr = m_spectral_obj.energy(m_spec_arr)
		mfcc_arr, m_delta_arr, m_deltadelta_arr = m_xxcc_obj.xxcc_standard(
		    m_spec_arr,
		    m_energy_arr,
		    cc_num=mfcc_coeffs,
		    delta_window_length=9,
		    energy_type=af.type.CepstralEnergyType.IGNORE,
		    rectify_type=af.type.CepstralRectifyType.LOG
		)

		# compute spectral features
		s_obj = af.BFT(
		    num=Int64(stft_length/2+1),
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

		centroid_arr = s_spectral_obj.centroid(s_spec_arr)
		crest_arr = s_spectral_obj.crest(s_spec_arr)
		decrease_arr = s_spectral_obj.decrease(s_spec_arr)
		entropy_arr = s_spectral_obj.entropy(s_spec_arr)
		flatness_arr = s_spectral_obj.flatness(s_spec_arr)
		flux_arr = s_spectral_obj.flux(s_spec_arr)
		kurtosis_arr = s_spectral_obj.kurtosis(s_spec_arr)
		rolloff_arr = s_spectral_obj.rolloff(s_spec_arr)
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
		    m_spec_arr[1:13, :],
		    mfcc_arr,
		    # m_delta_arr,
		    # m_deltadelta_arr,
		    centroid_arr',
		    crest_arr',
		    decrease_arr',
		    entropy_arr',
		    flatness_arr',
		    flux_arr',
		    kurtosis_arr',
		    rolloff_arr',
		    # skewness_arr',
		    slope_arr',
		    # spread_arr',
		    fre_arr',
		)...)

	else
		error("Unknown profile set: ", profile, ".")
	end
end
