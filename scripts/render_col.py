import numpy as np

## render column of image
def render_col(height, y_offset, thickness):
    col = np.full((height,), 255, dtype=np.uint8)
    centre = height // 2 + int(round(y_offset))
    radius = max(2, int(3*thickness))
    denom = 2 * (thickness ** 2) + 1e-6

    for dy in range(-radius, radius + 1):
        yy = centre + dy
        if 0 <= yy < height:
            intensity = np.exp(-(dy**2) / denom)
            intensity *= np.random.uniform(0.85, 1.0) # paper grain
            col[yy] = min(col[yy], int(255 - 175 * intensity))
    
    return col