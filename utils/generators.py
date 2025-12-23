import numpy as np
from scipy.ndimage import gaussian_filter1d

class ARGenerator:
    def __init__(self, phi, noise_std, seed=None):
        self.phi = phi
        self.noise_std = noise_std
        self.p = len(phi)

        if seed is None:
            self.state = np.zeros(self.p, dtype=np.float64)
        else:
            self.state = seed[-1:-(self.p+1):-1].copy()
    
    def step(self):
        epsilon = np.random.normal(0.0, self.noise_std)
        # compute next value
        x = np.dot(self.phi, self.state) + epsilon
        self.state[1:] = self.state[:-1]
        self.state[0] = x
        return x
    
class EnvelopeGenerator:
    def __init__(self, envelopes):
        self.envelopes = envelopes
        self._new_chunk()

    def _new_chunk(self):
        env = self.envelopes[np.random.randint(len(self.envelopes))]
        self.env = env
        self.idx = 0
    
    def step(self):
        if self.idx >= len(self.env):
            self._new_chunk()
        value = self.env[self.idx]
        self.idx += 1
        return value
    
class ThicknessGenerator:
    def __init__(self, mean=2.5, jitter=0.2, smooth=50):
        raw = mean + np.random.normal(0.0, jitter, size=500)
        self.thickness = gaussian_filter1d(raw, sigma=smooth)
        self.thickness = np.clip(self.thickness, 0.5, None)
        self.idx = 0

    def step(self):
        value = float(self.thickness[self.idx])
        self.idx = (self.idx + 1) % len(self.thickness)
        return value