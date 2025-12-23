import numpy as np

class ScrollBuffer:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.img = np.full((height, width), 255, dtype=np.uint8)

    def append_col(self, col):
        self.img[:, :-1] = self.img[:, 1:]
        self.img[:, -1] = col