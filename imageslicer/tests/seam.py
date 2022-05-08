import unittest
from .. import imageslicer
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tqdm

THRESHOLD = 100


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
        raw = imageio.imread("/home/mathmada/Downloads/9300255_Pferdestudien.jpg")

        raw = crop_image_v2(raw, 230)

        matplotlib.use("TkAgg")
        plt.imshow(raw, interpolation='nearest')
        plt.show()

        images, rest = imageslicer.core.slice(raw,
                                              300, 150)

        for i, img in enumerate(images):
            imageio.imwrite(f"./{i}.png", img)

        imageio.imwrite("./rest.png", rest)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
