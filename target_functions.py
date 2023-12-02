import autograd.numpy as autonp
import numpy as np


def rastrigin(x, A=10.0, b=1.0):
    n = x.shape[0]
    pi = np.pi

    x_square_part = autonp.square(x)
    x_cos_part = autonp.cos(2 * b * pi * x)

    res = A * n + autonp.sum(x_square_part - A * x_cos_part)

    return res
