import os
import numpy as np

DATA_DIR = "output/processed"

signals = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith("_signal.npy"):
        path = os.path.join(DATA_DIR, fname)
        signal = np.load(path)
        signals.append(signal)

print(f"Loaded {len(signals)} signals for training.")

P = 30

# fit autoregressive model of order p
def fit_ar(signals, p):
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

    return phi, noise_std

phi, noise_std = fit_ar(signals, P)

print("AR coefficients:", phi)
print("Estimated noise standard deviation:", noise_std)