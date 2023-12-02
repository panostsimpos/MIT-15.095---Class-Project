# %%
import numpy
import target_functions
from matplotlib import pyplot as plt
import tools
from sklearn.tree import DecisionTreeRegressor

# %%

lower_bound = -10
upper_bound = 10
n_dim = 2
n_samples = 200
n_new_samples = 100
mu = 0.5  # probability of bernouli gradient descent
n_algorithm_iteration = 1000

# Initialize
x_samples = numpy.random.uniform(lower_bound, upper_bound, (n_samples, n_dim))
descent_results = []
for x_sample in x_samples:
    descent_results.append(tools.gradient_descent(target_functions.rastrigin, x_sample))
descent_results = numpy.array(descent_results)
descent_scores = tools.descent_results_to_scores(descent_results)

best_sample = descent_results[0]

# main loop
for i in range(n_algorithm_iteration):
    tree = DecisionTreeRegressor()
    tree.fit(x_samples, descent_scores)

    tree = tools.build_prob_tree(tree, x_samples)  # in place modification

    new_x_samples = []
    new_y_samples = []

    sample_is_optimal = False

    # Create new samples
    for j in range(n_new_samples):
        x_new = numpy.random.uniform(lower_bound, upper_bound, n_dim)
        is_a_good_sample = tools.sample_check(x_new, tree)

        if not is_a_good_sample:
            continue

        is_bernouli_gradient = tools.decide_bernouli_gradient(mu)

        if is_bernouli_gradient:
            is_a_good_point, new_descent_result = tools.bernouli_gradient_descent(
                target_functions.rastrigin, x_new, tree
            )
            if not is_a_good_point:
                continue
        else:
            new_descent_result = tools.gradient_descent(
                target_functions.rastrigin, x_new
            )

        if tools.is_optimal(new_descent_result, epsilon=1e-6):  # optimality check
            sample_is_optimal = True
            break
        if numpy.linalg.norm(
            target_functions.rastrigin(new_descent_result)
        ) < numpy.linalg.norm(target_functions.rastrigin(best_sample)):
            best_sample = new_descent_result  # update best sample

        new_x_samples.append(x_new)
        new_y_samples.append(new_descent_result)

    if sample_is_optimal:  # break main loop at optimality
        best_sample = new_descent_result
        break

    new_x_samples = numpy.array(new_x_samples)
    new_y_samples = numpy.array(new_y_samples)

    if len(new_x_samples) == 0:
        print("No new data generated in run " + str(i) + ".")
        continue
    # Add new samples to the old ones
    x_samples = numpy.vstack((x_samples, new_x_samples))
    descent_scores = numpy.concatenate(
        (descent_scores, tools.descent_results_to_scores(new_y_samples))
    )

results = target_functions.rastrigin(best_sample)
print("\n The best result is: " + str(results))

# %%
