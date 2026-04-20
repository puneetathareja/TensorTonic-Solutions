import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X, y = np.array(X), np.array(y)
    n_samples, n_features = X.shape

    def gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)
    best_gain, best_feat, best_thresh = -np.inf, None, None

    for feat in range(n_features):
        vals = np.sort(np.unique(X[:, feat]))
        thresholds = (vals[:-1] + vals[1:]) / 2  

        for thresh in thresholds:
            left_mask  = X[:, feat] <= thresh
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                continue

            n_l, n_r = left_mask.sum(), right_mask.sum()
            weighted_gini = (n_l / n_samples) * gini(y[left_mask]) + \
                            (n_r / n_samples) * gini(y[right_mask])

            gain = parent_gini - weighted_gini

            
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

    return [best_feat, best_thresh]