import os
from PIL import Image
import numpy as np

from utils.pen_stroke import generate_thickness, draw_col

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
    path = next_available_filename(base="generated_series", ext=".png", directory="output")
    img.save(path)
    print(f"Generated series image saved to {path}.")