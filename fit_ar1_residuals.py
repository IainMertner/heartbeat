import numpy as np
import json

from load_signals import load_signals

## compute residuals
def residuals_per_signal(signal, phi):
    p = len(phi)
    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) <= p:
        return np.array([])

    residuals = []
    for t in range(p, len(signal)):
        predicted = float(phi @ signal[t-p:t][::-1])
        residuals.append(signal[t] - predicted)

    return np.array(residuals)

def compute_residuals(signals, phi):
    all_residuals = []

    for signal in signals:
        res = residuals_per_signal(signal, phi)
        if len(res) > 0:
            all_residuals.append(res)

    return np.concatenate(all_residuals)

## fit AR(1) model to residuals
def fit_ar1(residuals):
    r0 = np.mean(residuals[:-1] * residuals[:-1])
    r1 = np.mean(residuals[:-1] * residuals[1:])
    alpha = r1 / r0
    sigma_eta = np.std(residuals[1:] - alpha * residuals[:-1])

    return alpha, sigma_eta


# load signals
signals = load_signals()

## load AR model

MODEL_PATH = "output/ar_model.json"

with open(MODEL_PATH, "r") as f:
    model = json.load(f)
    phi = np.array(model["phi"])
    noise_std = model["noise_std"]

# compute residuals
residuals = compute_residuals(signals, phi)
# fit AR(1) to residuals
alpha, sigma_eta = fit_ar1(residuals)

## save AR(1) model
model = {
    "alpha": float(alpha),
    "sigma_eta": float(sigma_eta)
    }

with open("output/ar1_residuals_model.json", "w") as f:
    json.dump(model, f, indent=2)