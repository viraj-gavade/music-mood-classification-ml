import numpy as np
import librosa

def extract_math_features(path, sr=22050, duration=5.0, offset=0.0):
    """Original feature extractor used during preprocessing."""
    y, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    if len(y) == 0:
        raise ValueError(f"Empty audio: {path}")

    rms = np.sqrt(np.mean(y ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mag = S.mean(axis=1)

    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-8)

    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean())

    mag_norm = mag / (mag.sum() + 1e-8)
    mag_norm = mag_norm[mag_norm > 0]
    entropy = -np.sum(mag_norm * np.log2(mag_norm + 1e-12))

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        f0 = f0[~np.isnan(f0)]
        pitch_var = float(np.var(f0)) if len(f0) > 0 else 0.0
    except:
        pitch_var = 0.0

    return {
        'rms': float(rms),
        'zcr': float(zcr),
        'centroid': float(centroid),
        'rolloff': float(rolloff),
        'entropy': float(entropy),
        'pitch_var': float(pitch_var),
    }


def extract_math_features_from_array(y, sr=22050):
    """Optimized version used during inference (no file reload)."""
    if len(y) == 0:
        raise ValueError("Empty audio slice")

    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mag = S.mean(axis=1)

    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-8))
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

    mag_norm = mag / (mag.sum() + 1e-8)
    mag_norm = mag_norm[mag_norm > 0]
    entropy = float(-np.sum(mag_norm * np.log2(mag_norm + 1e-12)))

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        f0 = f0[~np.isnan(f0)]
        pitch_var = float(np.var(f0)) if len(f0) > 0 else 0.0
    except:
        pitch_var = 0.0

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "rolloff": rolloff,
        "entropy": entropy,
        "pitch_var": pitch_var,
    }
