import os
import json
import numpy as np
from PIL import Image
from scipy import stats

### config

IMAGE_DIR = "raw/cropped"
OUTPUT_DIR = "output/processed"
EXTENSIONS = [".png"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

### helper functions

## load image and convert to greyscale
def load_greyscale(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)

## compute greyscale-weighted vertical centroid for each column
def extract_centroid_signal(grey):
    h, w = grey.shape
    weights = 255.0 - grey
    y_indices = np.arange(h, dtype=np.float32).reshape(-1, 1)
    col_sums = weights.sum(axis=0)
    col_sums[col_sums == 0] = np.nan  # avoid division by zero
    centroids = (weights * y_indices).sum(axis=0) / col_sums
    ## interpolate missing columns
    if np.any(np.isnan(centroids)):
        valid = ~np.isnan(centroids)
        centroids = np.interp(np.arange(w), np.where(valid)[0], centroids[valid])
    
    return centroids

## detrend signal using linear regression
def detrend(signal):
    x = np.arange(len(signal))
    slope, intercept, _, _, _, = stats.linregress(x, signal)
    trend = slope * x + intercept
    return signal - trend, slope, intercept

## normalise signal to zero mean and unit variance
def normalise(signal):
    mean = signal.mean()
    std = signal.std()
    if std < 1e-6:
        raise ValueError("Standard deviation too small for normalisation.")
    normalised = (signal - mean) / std
    return normalised, mean, std

### main processing loop

results = []

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.lower().endswith(tuple(EXTENSIONS)):
        continue

    path = os.path.join(IMAGE_DIR, fname)
    print(f"Processing {fname}...")

    grey = load_greyscale(path)
    centroid_signal = extract_centroid_signal(grey)
    detrended_signal, slope, intercept = detrend(centroid_signal)
    normalised_signal, mean, std = normalise(detrended_signal)

    base = os.path.splitext(fname)[0]
    npy_path = os.path.join(OUTPUT_DIR, f"{base}_signal.npy")
    np.save(npy_path, normalised_signal)

    metadata = {
        "filename": fname,
        "length": int(len(normalised_signal)),
        "detrend_slope": float(slope),
        "detrend_intercept": float(intercept),
        "normalise_mean": float(mean),
        "normalise_std": float(std)
    }

    json_path = os.path.join(OUTPUT_DIR, f"{base}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    results.append(metadata)

# save summary metadata
with open(os.path.join(OUTPUT_DIR, "metadata_summary.json"), "w") as f:
    json.dump(results, f, indent=2)