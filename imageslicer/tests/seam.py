import unittest
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



def crop_image_threaded(img: np.ndarray, tol: float = 0) -> np.ndarray:

    def worker(arr: list[int], idx: int, img: np.ndarray, op: BoundParam, tol: float) -> None:
        res = get_crop_mask(img, op, tol)
        arr[idx] = res

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
        t = threading.Thread(target=worker, args=(bounds, i, img, v, tol))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return img[bounds[2]:bounds[3], bounds[0]:bounds[1] + 1, :].copy()

def crop_image_v2(img: np.ndarray, tol: float = 0) -> np.ndarray:
    h, w, d = img.shape

    ops = [
        ((0, w, 1), (0, h, 1), False),
        ((w-1, 0, -1), (0, h, 1), False),
        ((0, h, 1), (0, w, 1), True),
        ((h - 1, 0, -1), (0, w, 1), True),
    ]

    bounds = []

    for v in tqdm.tqdm(ops, "Cropping operations"):
        w_for, h_for, idx_switch = v

        for i in range(*w_for):
            min_luma = np.infty
            for j in range(*h_for):
                r, g, b = img[j][i] if not idx_switch else img[i][j]
                luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                if luma < min_luma:
                    min_luma = luma

            if min_luma < tol:
                break

        bounds.append(i)

    return img[bounds[2]:bounds[3], bounds[0]:bounds[1] + 1, :].copy()


class SeamTestCase(unittest.TestCase):
    def test_energy(self):
        matplotlib.use("TkAgg")

        raw = imageio.imread("/home/mathmada/Downloads/9200170b_Inuoumono - Hundehatz.jpg")

        print(chr(27) + "[2J")
        raw = crop_image_threaded(raw, 230)

        images, rest = imageslicer.core.slice(raw,
                                              2500, 300, 0.2)

        for i, img in enumerate(images):
            imageio.imwrite(f"./{i}.png", img)

        imageio.imwrite("./rest.png", rest)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
