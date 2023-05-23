import numpy as np
import pkg_resources


def load_hydra_data():
    dataset_path = pkg_resources.resource_filename(__name__, 'hydra.npz')
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = data["labels"]
    return X, y, labels