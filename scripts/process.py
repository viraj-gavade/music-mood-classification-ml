import os
import numpy as np
import librosa
from pathlib import Path
from src.features import extract_math_features

# Paths
DATA_RAW = Path("data/raw/genres_original")
DATA_OUT = Path("data/processed")
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Audio processing params
TARGET_SR = 22050
WINDOW = 5.0   # seconds
HOP = 5.0      # non-overlapping

# Map GTZAN genres → moods
MOOD_MAP = {
    'blues': 'sad',
    'classical': 'calm',
    'country': 'sad',
    'disco': 'happy',
    'hiphop': 'energetic',
    'jazz': 'calm',
    'metal': 'energetic',
    'pop': 'happy',
    'reggae': 'happy',
    'rock': 'energetic'
}


def slice_audio(y, sr, window, hop):
    """Yield audio segments (y_seg) at fixed window length."""
    total_dur = librosa.get_duration(y=y, sr=sr)
    print(f"  Total duration: {total_dur:.2f}s")

    start = 0.0
    while start + window <= total_dur:
        print(f"    Slicing window: start={start:.2f}s → end={start + window:.2f}s")
        start_sample = int(start * sr)
        end_sample = int((start + window) * sr)
        yield start, y[start_sample:end_sample]
        start += hop


def process_file(audio_path: Path, genre: str, idx_start: int):
    """Process a single audio file → multiple slices."""
    print(f"\n  Processing file: {audio_path.name}")
    mood = MOOD_MAP.get(genre)
    
    if mood is None:
        print(f"    Skipped (no mood mapping).")
        return 0

    print(f"    → Mood label: {mood}")

    try:
        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"    Error loading file: {e}")
        return 0

    count = 0
    idx = idx_start

    for offset, y_slice in slice_audio(y, TARGET_SR, WINDOW, HOP):
        print(f"      Extracting slice at offset {offset:.2f}s")

        # Extract mathematical features
        try:
            math_feats = extract_math_features(
                path=audio_path,
                sr=TARGET_SR,
                duration=WINDOW,
                offset=offset
            )
        except Exception as e:
            print(f"      Error extracting math features: {e}")
            continue

        # Mel Spectrogram
        try:
            mel = librosa.feature.melspectrogram(
                y=y_slice,
                sr=TARGET_SR,
                n_mels=128
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
        except Exception as e:
            print(f"      Error extracting mel spectrogram: {e}")
            continue

        # Save slice
        outpath = DATA_OUT / f"sample_{idx}.npz"
        np.savez(outpath, math=math_feats, mel=mel_db, label=mood)

        print(f"      Saved → {outpath.name}")
        idx += 1
        count += 1

    print(f"  Finished {audio_path.name}: {count} slices")
    return count


def run():
    idx = 0
    total = 0

    print("\n========== Starting Preprocessing ==========\n")

    for genre in os.listdir(DATA_RAW):
        genre_path = DATA_RAW / genre
        if not genre_path.is_dir():
            continue

        print(f"\n*** Genre: {genre} ***")

        for audio_file in genre_path.iterdir():
            if audio_file.suffix.lower() not in [".wav", ".mp3", ".au"]:
                print(f"  Skipping non-audio file: {audio_file.name}")
                continue

            slices = process_file(audio_file, genre, idx)
            idx += slices
            total += slices

    print("\n========== Preprocessing Complete ==========")
    print(f"Total slices created: {total}\n")


if __name__ == "__main__":
    run()
