import os
from PIL import Image
import numpy as np

## render series as line image
def render_line_image(series, height=300, margin=20, stroke_halfwidth=1, bg_colour=255, fg_colour=0):
    series = np.asarray(series)
    width = len(series)

    img = Image.new("L", (width, height), color=bg_colour)
    pixels = img.load()

    y0 = height // 2

    for x in range(width):
        y = int(round(y0 + series[x]))

        for dy in range(-stroke_halfwidth, stroke_halfwidth + 1):
            yy = y + dy
            if margin <= yy < height - margin:
                pixels[x, yy] = fg_colour
    
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