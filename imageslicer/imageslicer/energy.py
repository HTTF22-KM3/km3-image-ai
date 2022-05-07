import numpy as np
from scipy.ndimage.filters import convolve


def calculate_image_energy(image: np.ndarray) -> float:
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ])

    filter_dv = np.stack([filter_dv] * 3, axis=2)

    image = image.astype("float32")
    convolved = np.absolute(convolve(image, filter_du)) + np.absolute(convolve(image, filter_dv))

    energy_map = convolved.sum(axis=2)

    return energy_map
