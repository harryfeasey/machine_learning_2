import numpy as np
import math


def logistic_regression(xs, ys, alpha, max_batches):

    xs = np.insert(xs, 0, np.ones(xs.shape[0]), axis=1)

    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    theta = np.zeros(xs.shape[1])

    # θj := θj + α(y (i) − hθ(x^(i)) xj^(i)
    i = 0
    for _ in range(max_batches):
        for x_i, y_i in zip(xs, ys):
            theta = theta + alpha * (y_i - sigmoid(np.matmul(theta.T, x_i))) * x_i
            i += 1

    return lambda x: sigmoid(np.matmul(theta.T, np.insert(x, 0, [1])))


xs = np.array([1, 2, 3, 101, 102, 103]).reshape((-1, 1))
ys = np.array([0, 0, 0, 1, 1, 1])
model = logistic_regression(xs, ys, 0.05, 10000)
test_inputs = np.array([1.5, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101.8, 97]).reshape((-1, 1))

for test_input in test_inputs:
    print("{:.2f}".format(np.array(model(test_input)).item()))