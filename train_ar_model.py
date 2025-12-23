import json
import numpy as np

from load_signals import load_signals

signals = load_signals()

P = 50  # AR model order

# fit autoregressive model of order p
def train_ar_model(signals, p):
    X = []
    y = []

    for signal in signals:
        if len(signal) <= p:
            continue

        for t in range(p, len(signal)):
            X.append(signal[t-p:t][::-1])
            y.append(signal[t])
        
    X = np.array(X)
    y = np.array(y)

    # solve least squares y = X @ phi
    phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ phi
    noise_std = residuals.std()

    model = {
        "phi": phi.tolist()
        }

    with open("output/ar_model.json", "w") as f:
        json.dump(model, f, indent=2)

    return model