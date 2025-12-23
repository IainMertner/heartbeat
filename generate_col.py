import numpy as np

from visualise import render_col

## generate single column of image
class ColumnGenerator:
    def __init__(self, ar_gen, env_gen, thick_gen, height=300, amp=10.0, max_step=2.0, max_accel=5.0):
        self.ar_gen = ar_gen
        self.env_gen = env_gen
        self.thick_gen = thick_gen
        self.height = height
        self.amp = amp
        self.max_step = max_step
        self.max_accel = max_accel
        self.prev_y = 0.0
        self.prev_v = 0.0

    def step(self):
        carrier = self.ar_gen.step()
        envelope = self.env_gen.step()
        # calculate raw y
        raw_y = self.amp * carrier * envelope
        # velocity limiting
        v = raw_y - self.prev_y
        v = np.clip(v, -self.max_step, self.max_step)
        # acceleration limiting
        a = v - self.prev_v
        a = np.clip(a, -self.max_accel, self.max_accel)
        # final dy and y
        v = self.prev_v + a
        y = self.prev_y + v
        self.prev_v = v
        self.prev_y = y

        thickness = self.thick_gen.step()
        return render_col(self.height, y, thickness)