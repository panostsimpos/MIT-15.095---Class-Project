# %%
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from autograd import grad
import tools


def build_prob_tree(tree, data):
    leaf_indices = tree.apply(data)
    unique_leaf_indices = np.unique(leaf_indices)

    normalization = np.sum(tree.tree_.value[unique_leaf_indices].flatten())

    for leaf_index in unique_leaf_indices:
        tree.tree_.value[leaf_index, 0, 0] = (
            tree.tree_.value[leaf_index, 0, 0] / normalization
        )
    return tree


def descent_results_to_scores(descent_results, power=1.0):
    return (1 / descent_results) ** (-power)


def gradient_step(target_function, x, learning_rate):
    dfdx = grad(target_function)
    x_new = x - learning_rate * dfdx(x)

    return x_new


def sample_check(x, tree):
    acceptance_prob = tree.predict(x.reshape(1, -1))[0]

    alpha = np.random.uniform(0, 1)
    if alpha <= acceptance_prob:
        return True
    else:
        return False


def decide_bernouli_gradient(mu):
    acceptance_prob = mu
    assert acceptance_prob <= 1.0 and acceptance_prob >= 0.0

    beta = np.random.uniform(0, 1)
    if beta <= acceptance_prob:
        return True
    else:
        return False


def is_optimal(x, epsilon=1e-6):
    if tools.rastrigin(x) < epsilon:
        return True
    else:
        return False


def bernouli_gradient_descent(
    target_function,
    x_start,
    model,
    learning_rate=0.01,
    pow_param=1.0 / 2.0,
    check_period=10,
):
    x = x_start

    for i in range(1000000):
        x_next = gradient_step(target_function, x, learning_rate)

        if i % check_period == 0:
            prob = model.predict(x_next.reshape(1, -1))[0]
            prob = prob**pow_param

            trial = np.random.binomial(1, prob)

            if trial == 0:
                return (False, x_next)

        if np.linalg.norm(x_next - x) < 1e-4:
            return (True, x_next)

        x = x_next

    return None


def gradient_descent(
    target_function,
    x_start,
    learning_rate=0.01,
):
    x = x_start

    for i in range(1000000):
        x_next = gradient_step(target_function, x, learning_rate)

        if np.linalg.norm(x_next - x) < 1e-4:
            return x_next

        x = x_next

    return None


# # %%

# X_data = np.array([[1], [2], [3], [4]])
# y_data = np.array([1, 2, 3, 4])

# tree = DecisionTreeRegressor()
# tree.fit(X_data, y_data)

# build_prob_tree(tree, X_data)

# for x in X_data:
#     print(tree.predict(x.reshape(1, -1))[0])
# %%
