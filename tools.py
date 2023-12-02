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
