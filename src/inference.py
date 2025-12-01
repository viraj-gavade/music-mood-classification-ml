import torch
import numpy as np
import librosa

from src.models import HybridModel
from src.features import extract_math_features_from_array

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MoodInference:
    def __init__(self, model_path="model.pth", sr=22050, window=5.0, hop=5.0):
        self.sr = sr
        self.window = window
        self.hop = hop
        self.labels = ["happy", "sad", "calm", "energetic"]

        # Load model
        self.model = HybridModel(num_classes=4).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        # Load math normalization stats from training
        self.math_mean = np.load("math_mean.npy")
        self.math_std = np.load("math_std.npy") + 1e-8

    def _slice_audio(self, y):
        total_dur = librosa.get_duration(y=y, sr=self.sr)
        slices, offsets = [], []
        t = 0.0

        while t + self.window <= total_dur:
            start = int(t * self.sr)
            end = int((t + self.window) * self.sr)
            slices.append(y[start:end])
            offsets.append(t)
            t += self.hop

        return slices, offsets

    def _mel_spectrogram(self, y_slice):
        # EXACT SAME PARAMS AS TRAINING
        mel = librosa.feature.melspectrogram(
            y=y_slice,
            sr=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def _pad_mel(self, mel, target_len=216):
        F, T = mel.shape
        if T < target_len:
            mel = np.pad(mel, ((0, 0), (0, target_len - T)), mode="constant")
        else:
            mel = mel[:, :target_len]
        return mel

    def _prepare_inputs(self, mel, math_dict):
        # Mel normalization
        mel = self._pad_mel(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Math vector (normalized same as training)
        math_vec = np.array(list(math_dict.values()), dtype=np.float32)
        math_vec = (math_vec - self.math_mean) / self.math_std
        math_tensor = torch.tensor(math_vec, dtype=torch.float32).unsqueeze(0)

        return mel_tensor.to(DEVICE), math_tensor.to(DEVICE)

    def predict_file(self, audio_path):
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        slices, offsets = self._slice_audio(y)

        slice_predictions = []
        slice_probs = []

        for y_slice in slices:
            math_feats = extract_math_features_from_array(y_slice, self.sr)
            mel = self._mel_spectrogram(y_slice)

            mel_tensor, math_tensor = self._prepare_inputs(mel, math_feats)

            with torch.no_grad():
                logits = self.model(mel_tensor, math_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            slice_predictions.append(self.labels[int(np.argmax(probs))])
            slice_probs.append(probs)

        # Average probabilities across slices
        final_probs = np.mean(slice_probs, axis=0)
        final_label = self.labels[int(np.argmax(final_probs))]

        return {
            "final_mood": final_label,
            "final_confidence": final_probs,
            "slice_moods": slice_predictions,
            "slice_confidences": slice_probs,
            "offsets": offsets
        }
