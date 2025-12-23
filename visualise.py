import os
from PIL import Image
import numpy as np

from utils.pen_stroke import generate_thickness, draw_col
from utils.resource_path import resource_path

## render series as line image
def render_line_image(series, height=300):
    width = len(series)

    img = Image.new("L", (width, height), color=255)
    pixels = img.load()

    y0 = height // 2
    thickness = generate_thickness(width)

    for x in range(width):
        y = int(round(y0 + series[x]))
        draw_col(pixels, x, y, thickness[x], height)
    
    return img

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

## find next available filename
def next_available_filename(base, ext, directory="."):
    i = 1
    while True:
        filename = f"{base}{i}{ext}"
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return path
        i += 1

## save image
def save_image(img):
    path = resource_path(next_available_filename(base="generated_series", ext=".png", directory="output"))
    img.save(path)
    print(f"Generated series image saved to {path}.")