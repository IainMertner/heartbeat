from scipy.ndimage import gaussian_filter1d
import numpy as np

def generate_thickness(length, mean=2.0, jitter=0.6, smooth=30):
    thickness = mean + np.random.normal(0.0, jitter, size=length)
    thickness = gaussian_filter1d(thickness, sigma=smooth)

    return np.clip(thickness, 0.5, None)

def draw_col(pixels, x, y_centre, thickness, height):
    for dy in range(int(-3*thickness), int(3*thickness)+1):
        yy = y_centre + dy
        if 0 <= yy < height:
            intensity = np.exp(-(dy**2) / (2 * (thickness**2)))
            intensity *= np.random.uniform(0.85, 1.0)
            pixels[x, yy] = min(pixels[x, yy], int(255 * (1 - intensity)))