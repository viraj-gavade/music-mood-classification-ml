# src/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MoodDataset(Dataset):
    """
    Loads mel spectrogram + math features from processed NPZ files.
    Supports:
      - mel normalization
      - math normalization
      - time masking
      - frequency masking
      - automatic padding
    """

    def __init__(self, root="data/processed", augment=False):
        super().__init__()
        self.root = root
        self.files = [os.path.join(root, f) 
                       for f in os.listdir(root) if f.endswith(".npz")]
        self.augment = augment

        # Label encoding
        self.label_map = {
            "happy": 0,
            "sad": 1,
            "calm": 2,
            "energetic": 3
        }

        # Precompute math feature normalization mean/std
        self.math_mean, self.math_std = self.compute_math_stats()

    def compute_math_stats(self):
        """Compute global mean and std for math features."""
        feats = []
        for path in self.files:
            d = np.load(path, allow_pickle=True)
            math = d["math"].item()
            arr = np.array(list(math.values()), dtype=np.float32)
            feats.append(arr)

        feats = np.stack(feats)
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-8
        return mean, std

    def __len__(self):
        return len(self.files)

    def pad_mel(self, mel, target_len=216):
        """
        Pad or trim mel to fixed time dimension (T=216 approx for 5s).
        Ensures NN compatible batch shapes.
        """
        F, T = mel.shape

        if T == target_len:
            return mel

        if T < target_len:
            pad_amt = target_len - T
            mel = np.pad(mel, ((0, 0), (0, pad_amt)), mode="constant")
        else:
            mel = mel[:, :target_len]
        return mel

    def time_mask(self, mel, max_mask=20):
        """Randomly zero out time segments (SpecAug)."""
        if mel.shape[1] <= max_mask:
            return mel
        t = np.random.randint(0, mel.shape[1] - max_mask)
        mel[:, t:t+max_mask] = mel[:, t:t+max_mask] * 0
        return mel

    def freq_mask(self, mel, max_mask=8):
        """Randomly zero out frequency bands (SpecAug)."""
        if mel.shape[0] <= max_mask:
            return mel
        f = np.random.randint(0, mel.shape[0] - max_mask)
        mel[f:f+max_mask, :] = mel[f:f+max_mask, :] * 0
        return mel

    def normalize_mel(self, mel):
        """Per-sample standardization."""
        mean = mel.mean()
        std = mel.std() + 1e-8
        return (mel - mean) / std

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)

        # Extract features
        mel = d["mel"]
        math = d["math"].item()

        # FIX: convert numpy array to string
        label_str = str(d["label"])
        label = self.label_map[label_str]

        # Convert math dict → vector
        math_vec = np.array(list(math.values()), dtype=np.float32)

        # Normalize math features
        math_vec = (math_vec - self.math_mean) / self.math_std

        # Pad mel to 128×216
        mel = self.pad_mel(mel)

        # Augmentation (train only)
        if self.augment:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)

        # Normalize mel
        mel = self.normalize_mel(mel)

        # Convert to tensors
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        math_vec = torch.tensor(math_vec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return mel, math_vec, label
