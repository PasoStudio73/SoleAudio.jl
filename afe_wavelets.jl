using Audio911
# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
nan_replacer!(x::AbstractArray{Float64}) = replace!(x, NaN => 0.0)

# ---------------------------------------------------------------------------- #
#                       audio911 audio features extractor                      #
# ---------------------------------------------------------------------------- #
function audio911_extractor(
    # audio module
    wavfile::Union{String, AbstractVector{Float64}};
    sr::Int64=8000,
    norm::Bool=true,
    speech_detection::Bool=false,
    # continuous wavelets filterbank module
	wavelet::Symbol = :bump,                       # :morse, :morlet, :bump
	morse_params::Tuple{Int64, Int64} = (3, 60),
	vpo::Int64 = 10,
    freq_range::Tuple{Int64, Int64} = (80, round(Int, sr / 2)),
    # continuous wavelets transform module
    cwt_norm::Symbol = :power, # :power, :magnitude, :pow2mag
    db_scale::Bool = false,
    # mfcc module
    ncoeffs::Union{Nothing, Int64}=nothing,
    rectification::Symbol=:log,             # :log, :cubic_root
    dither::Bool=true,
    # # deltas module
    # d_length = 9,
    # d_matrix = :transposed,                 # :standard, :transposed
    # f0 module
    method::Symbol=:nfc,
    f0_range::Tuple{Int64, Int64}=(50, 400),
    # spectral features module
    spect_range::Union{Tuple{Int64, Int64}, Nothing}=nothing,
)
    # audio module
    audio = load_audio(
        file=wavfile, 
        sr=sr, 
        norm=norm,
    );
    if speech_detection
        audio = speech_detector(audio=audio);
    end

    # continuous wavelets filterbank module
    cwtfb = get_cwtfb(
        audio=audio,
        wavelet=wavelet,
        morse_params=morse_params,
        vpo=vpo,
        freq_range=freq_range
    );

    # continuous wavelets transform module
    cwt = get_cwt(
        audio=audio, 
        fbank=cwtfb,
        norm=cwt_norm,
        db_scale=db_scale
    );

    # mfcc module
    if isnothing(ncoeffs)
        ncoeffs = round(Int, size(cwt.spec, 1) / 2)
    end
    mfcc = get_mfcc(
        source=cwt,
        ncoeffs=ncoeffs,
        rectification=rectification,
        dither=dither,
    );

    # # deltas module
    # deltas = get_deltas(
    #     source=cwt,
    #     d_length=d_length,
    #     d_matrix=d_matrix
    # );

    # f0 module
    # TODO resolve current hack in f0 algorithm.
    cwtf0 = get_stft(
        audio=audio, 
        stft_length=256,
        win_type = (:hann, :periodic),
        win_length=256,
        overlap_length = 255,
        norm = :power, # :none, :power, :magnitude, :pow2mag
    );

    f0 = get_f0(
        source=cwtf0,
        method=method,
        freq_range=f0_range
    );
    #[hack]
    f0.f0 = vcat(f0.f0, zeros(length(audio.data)-length(f0.f0)))

    # spectral features module
    if isnothing(spect_range)
        spect_range = freq_range
    end
    spect = get_spectrals(
        source=cwt,
        freq_range=spect_range
    );

    return hcat(
        cwt.spec',
        mfcc.mfcc',
        f0.f0,
        spect.centroid,
        spect.crest,
        spect.entropy,
        spect.flatness,
        spect.flux,
        spect.kurtosis,
        spect.rolloff,
        spect.skewness,
        spect.decrease,
        spect.slope,
        spect.spread
    );
end