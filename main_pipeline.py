# %%
import numpy
import scipy
import target_functions
from matplotlib import pyplot as plt
import tools
import sklearn


# %%

# Define the domain

lower_bound = -10
upper_bound = 10
n_dim = 3
n_grid = 10
grid = numpy.meshgrid(*[numpy.linspace(lower_bound, upper_bound, n_grid)] * n_dim)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# %%


# Sample rows as data

n_samples = 100
x_samples = numpy.random.uniform(lower_bound, upper_bound, (n_samples, n_dim))

# Do gradient descent from each sample point

descent_results = []
for x_sample in x_samples:
    descent_results.append(tools.gradient_descent(target_functions.rastrigen, x_sample))

# Train CART to predict the descent results from the sample points

tree = sklearn.tree.DecisionTreeRegressor()
tree.fit(x_samples, descent_results)


# %%
