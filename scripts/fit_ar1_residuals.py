import numpy as np
import json

from scripts.utils.resource_path import resource_path

def compute_residuals(signals, phi):
    all_residuals = []
    p = len(phi)
    # compute residuals for each signal
    for signal in signals:
        signal = np.asarray(signal, dtype=np.float64)
        # skip short signals
        if len(signal) <= p:
            res = []
            continue
        # compute residuals
        res = []
        for t in range(p, len(signal)):
            predicted = float(phi @ signal[t-p:t][::-1])
            res.append(signal[t] - predicted)

        if len(res) > 0:
            all_residuals.append(res)

    return np.concatenate(all_residuals)

## fit AR(1) model to residuals
def fit_ar1_residuals(residuals):
    r0 = np.mean(residuals[:-1] * residuals[:-1])
    r1 = np.mean(residuals[:-1] * residuals[1:])
    
    alpha = r1 / r0
    sigma_eta = np.std(residuals[1:] - alpha * residuals[:-1])

    # save AR(1) model
    model = {
        "alpha": float(alpha),
        "sigma_eta": float(sigma_eta)
        }

    with open(resource_path("output/ar1_residuals_model.json"), "w") as f:
        json.dump(model, f, indent=2)
    
    return model