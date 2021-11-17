import numpy as np


def rotation_mat2d(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [[c, -s],
         [s,  c]]
    )
