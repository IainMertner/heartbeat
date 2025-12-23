import numpy as np
import pygame
import time

from utils.load_signals import load_signals
from utils.scroll_buffer import ScrollBuffer
from utils.get_sigma import get_sigma
from utils.generators import ARGenerator, EnvelopeGenerator, ThicknessGenerator
from train_ar_model import train_ar_model
from utils.envelopes import extract_envelope
from generate_incremental import generate_col

signals = load_signals()

P = 50
HEIGHT = 300
WIDTH = 800

# get amplitude from training data
amp = get_sigma()

## fit autoregressive model of order p
ar_model = train_ar_model(signals, P)
phi = np.array(ar_model["phi"])
noise_std = ar_model["noise_std"]

# get envelopes
envelopes = [extract_envelope(signal, sigma=130) for signal in signals]

## scrolling buffer to hold generated image
buffer = ScrollBuffer(HEIGHT, WIDTH)

## generators
ar_gen = ARGenerator(phi, noise_std)
env_gen = EnvelopeGenerator(envelopes)
thick_gen = ThicknessGenerator()

### run generation loop and render
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

running = True
## main loop
while running:
    for event in pygame.event.get():
        # quit event
        if event.type == pygame.QUIT:
            running = False
    # generate next column
    col = generate_col(ar_gen, env_gen, thick_gen, HEIGHT, amp)
    # append to buffer
    buffer.append_col(col)
    # convert to rgb
    rgb = np.stack([buffer.img]*3, axis=-1)
    # render to screen
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
    screen.blit(surf, (0,0))
    pygame.display.flip()
    # frame rate limit
    clock.tick(300)

pygame.quit()