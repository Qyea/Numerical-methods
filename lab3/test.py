from math import sqrt, e, log


# w_n+1 = 35/8 * x ^ 4 - 30/8 * x^2 + 3/8
exact =  (5 * log(5) - 11 * log(2))/15
k = 4
a, b = 0, 1
points = [-sqrt((1 / 35) * (15 - 2 * sqrt(30))),
           sqrt((1 / 35) * (15 - 2 * sqrt(30))),
          -sqrt((1 / 35) * (15 + 2 * sqrt(30))),
           sqrt((1 / 35) * (15 + 2 * sqrt(30)))]
A_coefs = [(1 / 36) * (18 + sqrt(30)),
           (1 / 36) * (18 + sqrt(30)),
           (1 / 36) * (18 - sqrt(30)),
           (1 / 36) * (18 - sqrt(30))]


def f(x):
    return (log(1 + x) / (3 * x + 2)**2)


def transform_points():
    return [(a + b) / 2 - 0.5 * (b - a) * t for t in points]


def nast():
    x = transform_points()
    result = 0
    for i in range(k):
        result += f(x[i]) * A_coefs[i]
    return result


print(exact)
print(nast())
