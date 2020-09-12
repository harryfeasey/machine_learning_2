

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


data = [(1, 4), (2, 7), (3, 10)]
m, c = linear_regression_2d(data)
print(m, c)
print(4 * m + c)