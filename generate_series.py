import numpy as np

from utils.envelopes import extract_envelope, sample_envelope

### generate synthetic time series from trained AR model
def generate_series(phi, alpha, sigma_eta, length=2000, seed=None):
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

### process generated series
def process_series(signals, generated_series_norm, sigma):
    # get envelopes
    envelopes = [extract_envelope(signal, sigma=130) for signal in signals]
    # apply envelope
    env = sample_envelope(envelopes, length=len(generated_series_norm))
    generated_series_mod = generated_series_norm * env
    # print stats
    print("generated_norm std:", generated_series_norm.std())
    print("generated_norm min/max:", generated_series_norm.min(), generated_series_norm.max())
    # scale by random normalisation std
    generated_series = generated_series_mod * sigma

    ## save generated series
    OUTPUT_PATH = "output/generated_series.npy"
    np.save(OUTPUT_PATH, generated_series)
    print(f"Generated synthetic series saved to {OUTPUT_PATH}.")

    return generated_series