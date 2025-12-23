import numpy as np

from scripts.utils.load_signals import load_signals
from scripts.utils.get_sigma import get_sigma
from scripts.train_ar_model import train_ar_model
from scripts.fit_ar1_residuals import compute_residuals, fit_ar1_residuals
from scripts.generate_series import generate_series, process_series
from scripts.visualise import render_line_image, next_available_filename, save_image

signals = load_signals()

P = 50
length = 2000

## fit autoregressive model of order p
ar_model = train_ar_model(signals, P)
phi = np.array(ar_model["phi"])

## compute residuals
residuals = compute_residuals(signals, phi)
## fit AR(1) model to residuals
ar1_residuals_model = fit_ar1_residuals(residuals)
alpha = np.array(ar1_residuals_model["alpha"])
sigma_eta = ar1_residuals_model["sigma_eta"]

sigma = get_sigma()

## generate synthetic series
generated_series_norm = generate_series(phi, alpha, sigma_eta, length=length)
## process generated series
generated_series = process_series(signals, generated_series_norm, sigma)

## visualise generated series
img = render_line_image(generated_series)
save_image(img)