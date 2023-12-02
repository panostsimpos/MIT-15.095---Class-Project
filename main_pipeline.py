# %%
import numpy
import scipy
import target_functions
from matplotlib import pyplot as plt
import tools
import sklearn

# %%
from target_functions import rastrigin
from tools import gradient_descent
import numpy as np

lower_bound = -10
upper_bound = 10
n_dim = 3
n_grid = 10
n_samples = 1000
n_new_samples = 100

# grid = numpy.meshgrid(*[numpy.linspace(lower_bound, upper_bound, n_grid)] * n_dim)

# Sample rows as data
x_samples = numpy.random.uniform(lower_bound, upper_bound, (n_samples, n_dim))

# Do gradient descent from each sample point

descent_results = []
for x_sample in x_samples:
    descent_results.append(tools.gradient_descent(target_functions.rastrigen, x_sample))

# Train CART to predict the descent results from the sample points

tree = sklearn.tree.DecisionTreeRegressor()
descent_scores = tools.descent_results_to_scores(descent_results)

tree.fit(x_samples, descent_scores)

probabilities = tools.predict_proba_global(tree, x_sample)
proba_tree = tools.build_prob_tree(tree, probabilities)

new_x_samples = []
new_y_sampels = []


for i in range(n_new_samples):
    x_new = numpy.random.uniform(lower_bound, upper_bound, n_dim)
    is_a_good_sample = tools.sample_check(x_new, probabilities, tree)

    if not is_a_good_sample:
        continue

    is_bernouli_gradient = tools.decide_bernouli_gradient(x_new, probabilities, tree)

    if is_bernouli_gradient:
        new_x_samples.append(x_new)
        new_y_sampels.append(
            tools.bernouli_gradient_descent(
                target_functions.rastrigen, x_new, proba_tree
            )
        )
