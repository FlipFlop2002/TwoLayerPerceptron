import numpy as np

def train_eval_test_split(X, Y):
    """
    Returns data in order: X_train, Y_train, X_eval, Y_eval, X_test, Y_test
    """
    X_train = X[0::2]
    Y_train = Y[0::2]

    X_rest = X[1::2]
    Y_rest = Y[1::2]

    X_eval = X_rest[0::2]
    Y_eval = Y_rest[0::2]
    X_test = X_rest[1::2]
    Y_test = Y_rest[1::2]

    return X_train, Y_train, X_eval, Y_eval, X_test, Y_test

def create_dataset(X, Y):
    data = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        data.append([x, y])
    dataset = np.array(data)
    return dataset


def fx(mi, b, x_min, x_max, num_samples):
    X = np.linspace(x_min, x_max, num_samples)
    Y = []
    for x in X:
        y = (1 / 2 * b) * np.exp(-1*(np.abs(x-mi) / b))
        Y.append(y)
    return X, np.array(Y)

def fx2(mi, b, x_min, x_max, num_samples):
    X = np.linspace(x_min, x_max, num_samples)
    Y = []
    for x in X:
        y = x**2 + 2
        Y.append(y)
    return X, np.array(Y)

def fx3(mi, b, x_min, x_max, num_samples):
    X = np.linspace(x_min, x_max, num_samples)
    Y = []
    for x in X:
        y = 0.5 * (np.tanh(x) + 1)
        Y.append(y)
    return X, np.array(Y)

def fx4(mi, b, x_min, x_max, num_samples):
    X = np.linspace(x_min, x_max, num_samples)
    Y = []
    for x in X:
        y = x**3 + x**2 - x -1
        Y.append(y)
    return X, np.array(Y)


class MSE_meter():
    def __init__(self):
        self.mse = 0
        self.e_values = np.array([])

    def update(self, true, pred):
        self.e_values = np.append(self.e_values, true-pred)
        self.mse = np.sum(np.square(self.e_values)) / len(self.e_values)

    def reset(self):
        self.mse = 0
        self.e_values = np.array([])


def calc_mse(Y, Y_pred):
    return np.sum(np.square(Y - Y_pred)) / len(Y)

def calc_mae(Y, Y_pred):
    return np.sum(np.abs(Y - Y_pred)) / len(Y)


# X, Y = fx(0, 1, -8, 8, 200)
# print(type(X), X)
# print(type(Y), Y)
# print(X.shape, Y.shape)