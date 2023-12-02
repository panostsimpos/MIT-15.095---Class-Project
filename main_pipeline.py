# %%
import numpy
import target_functions
from matplotlib import pyplot as plt
import tools
import sklearn

lower_bound = -10
upper_bound = 10
n_dim = 2
n_samples = 10
n_new_samples = 5
mu = 0.5  # probability of bernouli gradient descent
n_algorithm_iteration = 10

# n_grid = 10
# grid = numpy.meshgrid(*[numpy.linspace(lower_bound, upper_bound, n_grid)] * n_dim)

# Initialize
x_samples = numpy.random.uniform(lower_bound, upper_bound, (n_samples, n_dim))
descent_results = []
for x_sample in x_samples:
    descent_results.append(tools.gradient_descent(target_functions.rastrigin, x_sample))
descent_results = numpy.array(descent_results)
descent_scores = tools.descent_results_to_scores(descent_results)

# main loop
for i in range(n_algorithm_iteration):
    tree = sklearn.tree.DecisionTreeRegressor()
    tree.fit(x_samples, descent_scores)

    tree = tools.build_prob_tree(tree, x_samples)  # in place modification

    new_x_samples = []
    new_y_sampels = []

    sample_is_optimal = False

    # Create new samples
    for i in range(n_new_samples):
        x_new = numpy.random.uniform(lower_bound, upper_bound, n_dim)
        is_a_good_sample = tools.sample_check(x_new, tree)

        if not is_a_good_sample:
            continue

        is_bernouli_gradient = tools.decide_bernouli_gradient(mu)

        if is_bernouli_gradient:
            new_descent_result = tools.bernouli_gradient_descent(
                target_functions.rastrigin, x_new, tree
            )
        else:
            new_descent_result = tools.gradient_descent(
                target_functions.rastrigin, x_new
            )

        if tools.is_optimal(new_descent_result, epsilon=1e-6):  # optimality check
            sample_is_optimal = True
            break

        new_x_samples.append(x_new)
        new_y_sampels.append(new_descent_result)

    if sample_is_optimal:  # break main loop at optimality
        break

    new_x_samples = numpy.array(new_x_samples)
    new_y_sampels = numpy.array(new_y_sampels)

    # Add new samples to the old ones
    x_samples = numpy.vcat((x_samples, new_x_samples))
    y_samples = numpy.vcat(
        (descent_scores, tools.descent_results_to_scores(new_y_sampels))
    )
