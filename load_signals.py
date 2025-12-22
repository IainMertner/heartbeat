import os
import numpy as np

def load_signals():
    DATA_DIR = "output/processed"

    signals = []

    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_signal.npy"):
            path = os.path.join(DATA_DIR, fname)
            signal = np.load(path)
            signals.append(signal)

    print(f"Loaded {len(signals)} signals for training.")

    return signals