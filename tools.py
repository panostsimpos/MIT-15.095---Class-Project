# %%
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def gradient_descent(target_function, x_start, learning_rate, n_steps):
    pass


def predict_proba_global(tree, data):
    leaf_indices = tree.apply(data)
    unique_leaf_indices = np.unique(leaf_indices)
    normalization = np.sum(tree.tree_.value[unique_leaf_indices].flatten())

    return tree.predict(data) / normalization


def build_prob_tree(tree, data):
    pass


def descent_results_to_scores(descent_results):
    pass


def sample_from_proba_global(tree, probabilities):
    pass


# %%

import numpy as np
from autograd import grad
import tools


def gradient_step(target_function, x, learning_rate):
    dfdx = grad(target_function)
    x_new = x - learning_rate * dfdx(x)

    return x_new


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
