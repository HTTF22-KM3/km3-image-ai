import unittest

import cv2

from .. import imageslicer
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import threading

THRESHOLD = 100

BoundParam = list[tuple[int, int, int], tuple[int, int, int], bool]

def get_crop_mask(img: np.ndarray, op: BoundParam, tol: float) -> int:
    w_for, h_for, idx_switch = op

    t_id = threading.get_ident()

    for i in tqdm.tqdm(range(*w_for), f"Cropping thread {t_id}"):
        min_luma = np.infty
        for j in range(*h_for):
            r, g, b = img[j][i] if not idx_switch else img[i][j]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if luma < min_luma:
                min_luma = luma

        if min_luma < tol:
            break

    return i



def crop_image_threaded(img: np.ndarray, tol: float, downscale_factor: float) -> list[int]:

    def worker(arr: list[int], idx: int, img: np.ndarray, op: BoundParam, tol: float) -> None:
        res = get_crop_mask(img, op, tol)
        arr[idx] = res

    img = cv2.resize(img, (0, 0), fx=downscale_factor, fy=downscale_factor)

    h, w, d = img.shape

    ops = [
        ((0, w, 1), (0, h, 1), False),
        ((w-1, 0, -1), (0, h, 1), False),
        ((0, h, 1), (0, w, 1), True),
        ((h - 1, 0, -1), (0, w, 1), True),
    ]

    # 0 --> column start, 1 --> column end, 2 --> row start, 3 --> row end
    bounds = [0, 0, 0, 0]
    threads = []

    for i, v in enumerate(ops):
        t = threading.Thread(target=worker, args=(bounds, i, img, v, tol), daemon=True)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # return img[bounds[2]:bounds[3], bounds[0]:bounds[1] + 1, :].copy()
    return [int(b * (1 / downscale_factor)) for b in bounds]


class SeamTestCase(unittest.TestCase):
    def test_energy(self):
        matplotlib.use("TkAgg")

        raw = imageio.imread("/home/mathmada/Downloads/9200168_Yabusame - Ritterspiele (1).jpg")

        bounds = crop_image_threaded(raw, 230, 0.2)
        # bounds = [271, 34632, 47, 1509]
        raw = raw[bounds[2]:bounds[3], bounds[0]:bounds[1], :].copy()

        print("Starting slicing...")
        images, rest = imageslicer.core.slice(raw,
                                              1000, 100, 0.05)

        save_downscale = 0.2
        print(f"Saving {len(images)} images...")
        for i, img in tqdm.tqdm(enumerate(images), "Saving images..."):
            imageio.imwrite(f"./{i}.png", cv2.resize(img, (0, 0), fx=save_downscale, fy=save_downscale))

        imageio.imwrite("./rest.png", cv2.resize(rest, (0, 0), fx=save_downscale, fy=save_downscale))

        print("Done!")

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
