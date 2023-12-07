# %%
import numpy
import target_functions
from matplotlib import pyplot as plt
from tqdm import tqdm
import tools
from sklearn.tree import DecisionTreeRegressor
import time

# %%

lower_bound = -10
upper_bound = 10
n_dim = 5
n_new_samples, n_samples = 10000, 5000
mu = 1.0  # probability of bernouli gradient descent
n_algorithm_iteration = 200
max_leaves = 40

# Initialize
x_samples = numpy.random.uniform(lower_bound, upper_bound, (n_samples, n_dim))
descent_results = []
for x_sample in x_samples:
    descent_results.append(tools.gradient_descent(target_functions.rastrigin, x_sample))
descent_results = numpy.array(descent_results)
descent_scores = tools.descent_results_to_scores(descent_results, power=15.0)

best_sample = descent_results[0]

# main loop
histories = [x_samples]

start = time.time()
for i in tqdm(range(n_algorithm_iteration)):
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaves)
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

    # save history every 10 iterations
    if i % 10 == 0:
        histories.append(new_x_samples)

    # Add new samples to the old ones
    x_samples = numpy.vstack((x_samples, new_x_samples))
    descent_scores = numpy.concatenate(
        (descent_scores, tools.descent_results_to_scores(new_y_samples))
    )

    current_time = time.time()

    if current_time - start > 1000.0:
        print(best_sample)
        break

results = target_functions.rastrigin(best_sample)
print("\n The best result is: " + str(results))

# %%
plt.figure()
ax = plt.axes()
legend_labels = []  # Create an empty list to store legend labels

for i, history in enumerate(histories):
    if i % 2 == 0:
        continue
    ax.scatter(history[:, 0], history[:, 1])
    legend_labels.append("Iteration " + str(i))  # Store label for each iteration

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
plt.legend(legend_labels)  # Pass the list of labels to plt.legend()
plt.show()

# %%


plt.figure()
plt.hist2d(x_samples[:, 0], x_samples[:, 1], bins=100, cmap="Blues")
plt.colorbar()
plt.show()

# # Compute a histogram in 2D of the samples distribution
# for history in histories:
#     plt.figure()
#     plt.hist2d(history[:, 0], history[:, 1], bins=100, cmap="Blues")
#     plt.colorbar()
#     plt.show()

# %%
# print("Leaves:", tools.print_leaves(tree, x_samples))
# # print all x_samples in leaf with highest probability
# leaf_indices = tree.apply(x_samples)
# unique_leaf_indices = numpy.unique(leaf_indices)
# max_leaf_index = unique_leaf_indices[numpy.argmax(tree.tree_.value[:, 0, 0])]
# print("Max leaf index:", max_leaf_index)
# print("Max leaf value:", tree.tree_.value[max_leaf_index, 0, 0])
# print("Max leaf samples:")
# for i, leaf_index in enumerate(leaf_indices):
#     if leaf_index == max_leaf_index:
#         print(x_samples[i])
# %%
