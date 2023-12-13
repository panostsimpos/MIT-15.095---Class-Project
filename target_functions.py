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

# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np

# def rastrigin(x, y, A=10.0):
#     return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# # Make data.
# X = np.arange(-10, 10, 0.1)
# Y = np.arange(-10, 10, 0.1)
# X, Y = np.meshgrid(X, Y)
# Z = rastrigin(X, Y)

# # Create a contour plot.
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, cmap=cm.coolwarm)

# # Add labels and title.
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Contour Plot of the Rastrigin Function")

# # Add a color bar which maps values to colors.
# plt.colorbar(contour, shrink=0.5, aspect=5)

# plt.show()


# %%
