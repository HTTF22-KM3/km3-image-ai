import numpy as np
from scipy.ndimage.filters import convolve
import cv2

def calculate_image_energy(image: np.ndarray) -> np.ndarray:
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

def minimum_seam(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c, _ = image.shape
    energy_map = calculate_image_energy(image)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def slice(real_img: np.ndarray, part_width: int, skip_step: int = 5, downscale_factor: float = 0.5) -> tuple[list[np.ndarray], np.ndarray]:
    upscale_factor = 1.0 / downscale_factor

    img = cv2.resize(real_img, (0, 0), fx=downscale_factor, fy=downscale_factor)
    imgs: list[np.ndarray] = []

    iter = 0

    while len(img) != 0 and len(img[0]) > part_width * downscale_factor:
        iter += 1

        r, c, _ = img.shape

        M, backtrack = minimum_seam(img)

        # Create a (r, c) matrix filled with the value True
        # We'll be removing all pixels from the image which
        # have False later
        mask = np.ones((r, c), dtype=bool)

        # Find the position of the smallest element in the
        # last row of M
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            # Mark the pixels for deletion
            mask[i, j] = False
            j = backtrack[i, j]

        pos_min = np.infty
        for i in range(len(mask)):

            pos_false = 0
            for j in range(len(mask[i])):
                if not mask[i, j]:
                    pos_false = j
                    break

            if pos_false < pos_min:
                pos_min = pos_false

        left_img = real_img[0:, :int(pos_min * upscale_factor)]

        if len(left_img) == 0 or len(left_img[0]) == 0 or len(left_img[0]) < part_width:
            img = img[0:, int(skip_step * downscale_factor):]
            real_img = real_img[0:, skip_step:]
            continue


        imgs.append(left_img[0:, :part_width])
        img = img[0:, int(part_width * downscale_factor):]
        real_img = real_img[0:, part_width:]

    return (imgs, real_img)