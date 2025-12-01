import numpy as np
import os
from pathlib import Path

DATA_ROOT = Path("data/processed")

math_vectors = []

print("Scanning processed NPZ files...")

for file in DATA_ROOT.glob("*.npz"):
    data = np.load(file, allow_pickle=True)
    math_dict = data["math"].item()
    vec = np.array(list(math_dict.values()), dtype=np.float32)
    math_vectors.append(vec)

math_vectors = np.stack(math_vectors)

mean = math_vectors.mean(axis=0)
std = math_vectors.std(axis=0) + 1e-8

np.save("math_mean.npy", mean)
np.save("math_std.npy", std)

print("Saved math_mean.npy and math_std.npy")
print("Mean:", mean)
print("Std:", std)
