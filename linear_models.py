import numpy as np


def dot(x, y):

    total = 0
    for x_i, y_i in zip(x, y):
        total += x_i * y_i

    return total


def linear_regression_2d(data):

    x = []
    y = []

    for feature, response in data:
        x.append(feature)
        y.append(response)

    n = len(x)
    m = (n * dot(x, y) - (sum(x) * sum(y))) / (n * dot(x, x) - (sum(x) ** 2))
    c = (sum(y) - m*(sum(x))) / n
    return m, c


def linear_regression_(x, y, penalty=None):
    # (X.T * X + λI)^-1 X.T * y
    x_t_x = np.matmul(x.T, x)
    if penalty:
        identity = penalty * np.identity(x_t_x.shape[0])
        x_t_x = np.add(x_t_x, identity)
    inverted = np.linalg.inv(x_t_x)
    x_t_y = np.matmul(x.T, y)

    return np.matmul(inverted, x_t_y)


def linear_regression(xs, ys, basis_functions=None, penalty=0):

    if basis_functions or basis_functions == []:
        new_x = np.ones(xs.shape[0]).reshape(-1, 1)
        for function in basis_functions:
            x = [function(x) for x in xs]
            new_x = np.insert(new_x, new_x.shape[1], x, axis=1)

        return linear_regression_(new_x, ys, penalty)
    else:
        xs = np.insert(xs, 0, np.ones(xs.shape[0]), axis=1)
        return linear_regression_(xs, ys, penalty)


import numpy as np

xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)

print(linear_regression(xs, ys), end="\n\n")

with np.printoptions(precision=5, suppress=True):
    print(linear_regression(xs, ys, penalty=0.1))