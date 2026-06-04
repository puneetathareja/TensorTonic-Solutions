import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)

    results = []

    for x in X_test:

        scores = []

        for c in classes:

            X_c = X_train[y_train == c]

            prior = len(X_c) / len(X_train)

            p = (np.sum(X_c, axis=0) + 1) / (len(X_c) + 2)

            log_prob = np.log(prior)

            log_prob += np.sum(
                x * np.log(p) +
                (1 - x) * np.log(1 - p)
            )

            scores.append(log_prob)

        results.append(scores)

    return np.array(results)