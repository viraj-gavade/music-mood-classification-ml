import torch
import numpy as np
import librosa

from src.models import HybridModel
from src.features import extract_math_features_from_array

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MoodInference:
    def __init__(self, model_path="model.pth", sr=22050, window=5.0, hop=5.0):
        # EXTREME SPEED MODE
        self.sr = min(sr, 8000)   # Even lower sample rate for max speed
        self.window = window
        self.hop = hop
        self.labels = ["happy", "sad", "calm", "energetic"]

        # Load model with extreme optimization
        self.model = HybridModel(num_classes=4).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
        # Maximum speed optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Faster but less reproducible
        # Do not use .half() unless you are sure all inputs and weights are compatible
        # If you want FP16, ensure all inputs and weights are compatible and your GPU supports it
            
        # Skip torch.compile to avoid Triton dependency issues
        # Still get major speedup from other optimizations

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
        # ULTRA MINIMAL - Absolute minimum for speed
        mel = librosa.feature.melspectrogram(
            y=y_slice,
            sr=self.sr,
            n_fft=256,      # Absolute minimum FFT
            hop_length=64,  # Maximum hop for speed  
            n_mels=16       # Tiny mel count
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    
    def _get_math_only_prediction(self, y_slice):
        """Ultra-fast prediction using only mathematical features"""
        from src.features import extract_math_features_from_array
        
        # Extract only math features (very fast)
        math_feats = extract_math_features_from_array(y_slice, self.sr)
        math_vec = np.array(list(math_feats.values()), dtype=np.float32)
        math_vec = (math_vec - self.math_mean) / self.math_std
        
        # Simple heuristic-based prediction (instant)
        rms, zcr, centroid, rolloff, entropy, pitch_var = math_vec
        
        # Fast heuristic rules based on audio features
        if rms > 0.5 and zcr > 0.3:  # High energy + noisy
            return "energetic", [0.1, 0.1, 0.1, 0.7]
        elif centroid > 1000:  # Bright sound
            return "happy", [0.7, 0.1, 0.1, 0.1]  
        elif rms < 0.1:  # Low energy
            return "calm", [0.1, 0.1, 0.7, 0.1]
        else:  # Default to sad
            return "sad", [0.1, 0.7, 0.1, 0.1]
    
    def _batch_mel_spectrogram(self, audio_slices):
        """Compute mel spectrograms for multiple slices efficiently"""
        mel_specs = []
        for y_slice in audio_slices:
            mel_db = self._mel_spectrogram(y_slice)
            mel_specs.append(mel_db)
        return mel_specs
    
    def _batch_math_features(self, audio_slices):
        """Extract mathematical features for multiple slices efficiently"""
        math_features = []
        for y_slice in audio_slices:
            math_feat_dict = extract_math_features_from_array(y_slice, self.sr)
            # Convert dict to array in the same order as training
            math_feat = np.array(list(math_feat_dict.values()), dtype=np.float32)
            math_feat = (math_feat - self.math_mean) / self.math_std
            math_features.append(math_feat)
        return np.array(math_features)

    def _pad_mel(self, mel, target_len=20):  # Ultra-minimal padding
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
    
    def _prepare_inputs_batch(self, mel, math_vec_normalized):
        """Prepare inputs for batch processing where math features are already normalized"""
        # Mel normalization
        mel = self._pad_mel(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        # Always use float32 for input tensors
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        math_tensor = torch.tensor(math_vec_normalized, dtype=torch.float32).unsqueeze(0)
        return mel_tensor.to(DEVICE), math_tensor.to(DEVICE)

    def predict_file(self, audio_path, math_only=False):
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        slices, offsets = self._slice_audio(y)

        # ULTRA-FAST MODE: Math features only
        if math_only and len(slices) > 0:
            # Use only first slice for instant results
            y_slice = slices[0]
            mood, conf = self._get_math_only_prediction(y_slice)
            
            return {
                "final_mood": mood,
                "final_confidence": np.array(conf),
                "slice_moods": [mood],
                "slice_confidences": [conf],
                "offsets": [0.0],
                "window_size": self.window
            }

        # Standard batch processing for GPU acceleration
        if len(slices) > 0:
            # Extract all mel spectrograms and math features at once
            mel_specs = self._batch_mel_spectrogram(slices)
            math_features = self._batch_math_features(slices)
            
            # Prepare batch tensors
            mel_batch = []
            math_batch = []
            
            for i, (mel, math_feat) in enumerate(zip(mel_specs, math_features)):
                # Use batch version that doesn't double-normalize math features
                mel_tensor, math_tensor = self._prepare_inputs_batch(mel, math_feat)
                mel_batch.append(mel_tensor.squeeze(0))
                math_batch.append(math_tensor.squeeze(0))
            
            # Stack into batch tensors
            mel_batch_tensor = torch.stack(mel_batch).to(DEVICE)
            math_batch_tensor = torch.stack(math_batch).to(DEVICE)
            
            # Single batch inference (much faster than individual predictions)
            with torch.no_grad():
                # Ensure tensors are float32 and on correct device
                mel_batch_tensor = mel_batch_tensor.to(device=DEVICE, dtype=torch.float32)
                math_batch_tensor = math_batch_tensor.to(device=DEVICE, dtype=torch.float32)
                batch_logits = self.model(mel_batch_tensor, math_batch_tensor)
                batch_probs = torch.softmax(batch_logits, dim=1).cpu().numpy()
                batch_preds = torch.argmax(batch_logits, dim=1).cpu().numpy()
            
            slice_predictions = [self.labels[pred] for pred in batch_preds]
            slice_probs = batch_probs.tolist()
        else:
            slice_predictions = []
            slice_probs = []

        # Calculate final prediction from batch results
        if slice_probs:
            final_probs = np.mean(slice_probs, axis=0)
            final_label = self.labels[int(np.argmax(final_probs))]
        else:
            final_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Default uniform
            final_label = "unknown"

        return {
            "final_mood": final_label,
            "final_confidence": final_probs,
            "slice_moods": slice_predictions,
            "slice_confidences": slice_probs,
            "offsets": offsets
        }
