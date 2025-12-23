import numpy as np
from scipy.ndimage import gaussian_filter1d

def extract_envelope(signal, sigma=30):
    env = np.abs(signal)
    env = gaussian_filter1d(env, sigma=sigma)

    env /= np.mean(env)

    return env

def sample_envelope(envelopes, length):
    out = []
    while len(out) < length:
        env = envelopes[np.random.randint(len(envelopes))]
        out.extend(env.tolist())
        
    return np.array(out[:length], dtype=np.float64)