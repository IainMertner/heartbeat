import os
import json
import numpy as np

from load_signals import load_signals
from envelopes import extract_envelope, sample_envelope

## load AR(p) model
MODEL_PATH = "output/ar_model.json"
with open(MODEL_PATH, "r") as f:
    model = json.load(f)
    phi = np.array(model["phi"])

## load AR(1) residuals model
MODEL_PATH = "output/ar1_residuals_model.json"
with open(MODEL_PATH, "r") as f:
    model = json.load(f)
    alpha = np.array(model["alpha"])
    sigma_eta = model["sigma_eta"]

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

## generate synthetic time series from trained AR model
def generate_series(phi, alpha, sigma_eta, length=3000, seed=None):
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

    epsilon = 0.0

    ## generate series
    for t in range(length):
        epsilon = alpha * epsilon + np.random.normal(0.0, sigma_eta)
        # compute next value
        x = np.dot(phi, state) + epsilon
        state[1:] = state[:-1]
        state[0] = x
        out[t] = x
    
    return out

# get envelopes
signals = load_signals()
envelopes = [extract_envelope(signal, sigma=150) for signal in signals]

## generate synthetic series

generated_series_norm = generate_series(phi, alpha, sigma_eta, length=3000)
env = sample_envelope(envelopes, length=len(generated_series_norm))
generated_series_mod = generated_series_norm * env
print("generated_norm std:", generated_series_norm.std())
print("generated_norm min/max:", generated_series_norm.min(), generated_series_norm.max())
generated_series = generated_series_mod * sigma

## save generated series

OUTPUT_PATH = "output/generated_series.npy"
np.save(OUTPUT_PATH, generated_series)
print(f"Generated synthetic series saved to {OUTPUT_PATH}.")