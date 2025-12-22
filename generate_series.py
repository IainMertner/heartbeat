import os
import json
import numpy as np

## generate synthetic time series from trained AR model
def generate_series(phi, noise_std, length=3000, seed=None):
    # get model order
    p = len(phi)
    # initialise state
    if seed is None:
        state = np.zeros(p)
    else:
        seed = np.asarray(seed)
        state = seed[-p:] if len(seed) >= p else np.pad(seed, (p - len(seed), 0))
    
    out = np.zeros(length)
    ## generate series
    for t in range(length):
        # compute next value
        next_val = np.dot(phi, state) + np.random.normal(0, noise_std)
        # store output
        out[t] = next_val
        # update state
        state = np.roll(state, 1)
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

generated_series_norm = generate_series(phi, noise_std=sigma, length=3000)
generated_series = generated_series_norm * sigma

## save generated series

OUTPUT_PATH = "output/generated_series.npy"
np.save(OUTPUT_PATH, generated_series)
print(f"Generated synthetic series saved to {OUTPUT_PATH}.")