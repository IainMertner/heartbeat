import os
import json
import numpy as np

from utils.resource_path import resource_path

## load normalisation stds from metadata
def get_sigma():
    sigmas = []
    for fname in os.listdir(resource_path("output/processed")):
        if fname.endswith("_metadata.json"):
            path = os.path.join(resource_path("output/processed"), fname)
            with open(path, "r") as f:
                metadata = json.load(f)
                sigmas.append(metadata["normalise_std"])
                
    sigma = float(np.median(sigmas))

    return sigma