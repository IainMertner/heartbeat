import matplotlib.pyplot as plt
import numpy as np

generated_series = np.load("output/generated_series.npy")

plt.figure(figsize=(10,3))
plt.plot(generated_series, lw=1)
plt.title("Generated tremor signal")
plt.show()