import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]

    distances = ((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2).sum(axis=2)

    sorted_indices = np.argsort(distances, axis=1)

    result = -1 * np.ones((X_test.shape[0], k), dtype=int)

    valid_k = min(k, n_train)
    result[:, :valid_k] = sorted_indices[:, :valid_k]

    return result