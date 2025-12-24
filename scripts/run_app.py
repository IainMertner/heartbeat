import numpy as np
import pygame
import pygame._sdl2 as sdl2

from scripts.utils.load_signals import load_signals
from scripts.utils.scroll_buffer import ScrollBuffer
from scripts.utils.get_sigma import get_sigma
from scripts.utils.generators import ARGenerator, EnvelopeGenerator, ThicknessGenerator
from scripts.train_ar_model import train_ar_model
from scripts.utils.envelopes import extract_envelope
from scripts.generate_col import ColumnGenerator
from scripts.utils.slider import SpeedSlider

def run_app():

    signals = load_signals()

    P = 50
    HEIGHT = 300
    WIDTH = 1200
    MAX_STEP = 2.0
    MAX_ACCEL = 0.25

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
    col_gen = ColumnGenerator(ar_gen, env_gen, thick_gen, HEIGHT, amp, MAX_STEP)

    ### run generation loop and render
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Heartbeat")

    window = sdl2.Window.from_display_module()
    window.maximize()

    clock = pygame.time.Clock()
    
    # # scrolling speed control (columns per second)
    speed = 300.0        # initial speed
    accumulator = 0.0
    slider = SpeedSlider(
        x=WIDTH/2,
        y=HEIGHT*2,
        w=300,
        min_val=0,
        max_val=600,
        value=speed
    )

    running = True
    ## main loop
    while running:
        for event in pygame.event.get():
            # quit event
            if event.type == pygame.QUIT:
                running = False
            # resize event
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                buffer.resize_width(event.w)
            slider.handle_event(event)
        # time since last frame (seconds)
        dt = clock.get_time() / 1000.0
        # update speed from slider
        speed = slider.value
        accumulator += speed * dt
        while accumulator >= 1.0:
            # generate next column
            col = col_gen.step()
            # append to buffer
            buffer.append_col(col)
            accumulator -= 1.0
        # convert to rgb
        rgb = np.stack([buffer.img]*3, axis=-1)
        ## render to screen
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
        screen.fill((255, 255, 255))
        y_offset = (screen.get_height() - buffer.img.shape[0]) // 2
        screen.blit(surf, (0, y_offset))
        slider.draw(screen)
        pygame.display.flip()
        # frame rate limit
        clock.tick(60)

    pygame.quit()