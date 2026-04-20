import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left  = np.array(y_left)
    y_right = np.array(y_right)

    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    n_l, n_r = len(y_left), len(y_right)
    n = n_l + n_r

    if n == 0:
        return 0.0

    return (n_l / n) * gini(y_left) + (n_r / n) * gini(y_right)