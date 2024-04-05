import numpy as np


def split_data(X, y, p, r=None):
    rng = np.random.RandomState(123456)
    permutation = rng.permutation(X.shape[0])
    index_train = permutation[: int(p * X.shape[0])]
    index_test = permutation[int(p * X.shape[0]) :]
    X_train = X[index_train]
    X_test = X[index_test]
    y_train = y[index_train]
    y_test = y[index_test]

    if r is not None:
        r_train = r[index_train]
        r_test = r[index_test]
    else:
        r_train = np.ones(y_train.shape) / len(y_train)
        r_test = np.ones(y_test.shape) / len(y_test)

    return (X_train, y_train, r_train), (X_test, y_test, r_test)
