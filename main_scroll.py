import numpy as np

from utils.load_signals import load_signals
from utils.scroll_buffer import ScrollBuffer
from utils.get_sigma import get_sigma
from utils.generators import ARGenerator
from train_ar_model import train_ar_model
from utils.envelopes import extract_envelope
from generate_incremental import generate_col

signals = load_signals()

P = 50
HEIGHT = 300
WIDTH = 800

# get amplitude from training data
amp = get_sigma()
amp = 10

## fit autoregressive model of order p
ar_model = train_ar_model(signals, P)
phi = np.array(ar_model["phi"])
noise_std = ar_model["noise_std"]

# estimate carrier std once (warm-up)
warm = np.array([ARGenerator(phi, noise_std).step() for _ in range(3000)])
carrier_std = float(warm.std())

# get envelopes
envelopes = [extract_envelope(signal, sigma=130) for signal in signals]

## scrolling buffer to hold generated image
buffer = ScrollBuffer(HEIGHT, WIDTH)

## generate image incrementally by column
for _ in range(1000):
    col = generate_col(phi, noise_std, envelopes, HEIGHT, carrier_std, amp)
    buffer.append_col(col)

import matplotlib.pyplot as plt

plt.imshow(buffer.img, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.show()