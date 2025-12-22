import os
import json
import numpy as np

## generate synthetic time series from trained AR model
def generate_series(phi, noise_std, length=3000, seed=None):
    # get model order
    p = len(phi)
    # initialise state
    if seed is None:
        state = np.zeros(p, dtype=np.float64)
    else:
        seed = np.asarray(seed, dtype=np.float64).ravel()
        if len(seed) >= p:
            state = seed[-1:-(p+1):-1].copy()
        else:
            padded = np.pad(seed, (p - len(seed), 0))
            state = padded[-1:-(p+1):-1].copy()
    
    out = np.zeros(length, dtype=np.float64)

    ## generate series
    for t in range(length):
        # compute next value
        next_val = float(phi @ state) + np.random.normal(0.0, noise_std)
        # store output
        out[t] = next_val
        # update state
        state[1:] = state[:-1]
        state[0] = next_val
    
    return out

## load normalisation stds from metadata

DATA_DIR = "output/processed"

sigmas = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith("_metadata.json"):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r") as f:
            metadata = json.load(f)
            sigmas.append(metadata["normalise_std"])

sigma = np.random.choice(sigmas)

## load AR model

MODEL_PATH = "output/ar_model.json"

with open(MODEL_PATH, "r") as f:
    model = json.load(f)
    phi = np.array(model["phi"])
    noise_std = model["noise_std"]

## generate synthetic series

generated_series_norm = generate_series(phi, noise_std, length=3000)
print("generated_norm std:", generated_series_norm.std())
print("generated_norm min/max:", generated_series_norm.min(), generated_series_norm.max())
generated_series = generated_series_norm * sigma

## save generated series

OUTPUT_PATH = "output/generated_series.npy"
np.save(OUTPUT_PATH, generated_series)
print(f"Generated synthetic series saved to {OUTPUT_PATH}.")