import numpy as np

from visualise import render_col

## generate single column of image
class ColumnGenerator:
    def __init__(self, ar_gen, env_gen, thick_gen, height=300, amp=10.0, max_step=2.0):
        self.ar_gen = ar_gen
        self.env_gen = env_gen
        self.thick_gen = thick_gen
        self.height = height
        self.amp = amp
        self.max_step = max_step
        self.prev_y = 0.0

    def step(self):
        carrier = self.ar_gen.step()
        envelope = self.env_gen.step()

        raw_y = self.amp * carrier * envelope

        dy = raw_y - self.prev_y
        dy = np.clip(dy, -self.max_step, self.max_step)
        y = self.prev_y + dy
        self.prev_y = y

        thickness = self.thick_gen.step()
        return render_col(self.height, y, thickness)