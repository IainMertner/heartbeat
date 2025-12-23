import numpy as np

class ScrollBuffer:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.img = np.full((height, width), 255, dtype=np.uint8)

    def append_col(self, col):
        self.img[:, :-1] = self.img[:, 1:]
        self.img[:, -1] = col

    def resize_width(self, new_width):
        h, w = self.img.shape
        new = np.full((h, new_width), 255, dtype=np.uint8)
        # crop from the right (keep most recent data) if smaller
        if new_width <= w:
            new[:] = self.img[:, w - new_width : w]
        # otherwise, pad on the left
        else:
            new[:, new_width - w:] = self.img
        
        self.img = new