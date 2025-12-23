import os
import numpy as np

from utils.resource_path import resource_path

def load_signals():
    DATA_DIR = resource_path("output/processed")

    signals = []

    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_signal.npy"):
            path = os.path.join(DATA_DIR, fname)
            signal = np.load(path)
            signals.append(signal)

    print(f"Loaded {len(signals)} signals for training.")

    return signals