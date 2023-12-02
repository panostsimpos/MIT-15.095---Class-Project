# %%
import autograd.numpy as autonp
import numpy as np


def rastrigin(x, A=10.0, b=1.0):
    n = x.shape[0]
    pi = np.pi

    x_square_part = autonp.square(x)
    x_cos_part = autonp.cos(2 * b * pi * x)

    res = A * n + autonp.sum(x_square_part - A * x_cos_part)

    return res


# # find minimum
# from scipy.optimize import minimize
# x0 = np.random.uniform(-10, 10, 2)
# res = minimize(rastrigin, x0, method='nelder-mead',
#                options={'xatol': 1e-8, 'disp': True})
# print(res.x)
# print(rastrigin(res.x))


# # check non-negative

# for i in range(10000):
#     x = np.random.uniform(-10, 10, 2)
#     assert rastrigin(x) >= 0
# %%
