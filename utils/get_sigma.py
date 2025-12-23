import os
import json
import numpy as np

## load normalisation stds from metadata
def get_sigma():
    sigmas = []
    for fname in os.listdir("output/processed"):
        if fname.endswith("_metadata.json"):
            path = os.path.join("output/processed", fname)
            with open(path, "r") as f:
                metadata = json.load(f)
                sigmas.append(metadata["normalise_std"])
                
    sigma = float(np.median(sigmas))

    return sigma