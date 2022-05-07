import numpy as np
import energy

def minimum_seam(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c, _ = image.shape
    energy_map = energy.calculate_image_energy(image)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, )