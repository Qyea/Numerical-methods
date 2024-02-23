import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
from tabulate import tabulate

a = -3
b = 3

# Определение функции f1(x) = sin(cos(x))
def f1(x):
    return np.sin(np.cos(x))

# Определение функции f2(x) = ||x| - 1|
def f2(x):
    return np.abs(np.abs(x) - 1)

# Значения точек функции
def f_points(build_points, f):
    return [f(build_points[i]) for i in range(101)]

# Выбор узлов (равноотстоящие)
def choose_equidistant_nodes(n=100):
    # step = (b-a)/n => for(i){nodesArray+= a+step*i;}
    return np.linspace(a, b, n + 1) 

# (чебышёвские)
def choose_chebyshev_nodes(n):
    k = np.arange(n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    return nodes


def newton_interpolation(n, x, points, coefs):
    result = coefs[0]
    mult = 1
    for i in range(n):
        mult *= x - points[i]
        result += coefs[i + 1] * mult
    return result


def build_table(n, points, f):
    table = [[0 for j in range(n + 2)] for i in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = points[i]
    for i in range(n + 1):
        table[i][1] = f(points[i])

    for j in range(2, n + 2):
        for i in range(j - 1, n + 1):
            table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (points[i] - points[i - (j - 1)])

    return table


def P(n, f):
    points = choose_equidistant_nodes(n)
    table = build_table(n, points, f)

    coefs = []
    for i in range(1, n + 2):
        coefs.append(table[i - 1][i])

    build_points = choose_equidistant_nodes()
    result = []
    for i in range(101):
        result.append(newton_interpolation(n, build_points[i], points, coefs))

    return result


def C(n, f):
    points = choose_chebyshev_nodes(n)
    table = build_table(n, points, f)

    coefs = []
    for i in range(1, n + 2):
        coefs.append(table[i - 1][i])

    build_points = choose_equidistant_nodes()
    result = []
    for i in range(101):
        result.append(newton_interpolation(n, build_points[i], points, coefs))

    return result


def max_diff(n, f):
    points = choose_equidistant_nodes()
    p = P(n, f)
    c = C(n, f)
    F = f_points(points, f)
    p_max = max([abs(p[i] - F[i]) for i in range(101)])
    c_max = max([abs(c[i] - F[i]) for i in range(101)])
    return p_max, c_max


def error_table(f):
    d = {'n': pd.Series([i for i in range(3, 31)]),
         'p_max': pd.Series([round(Decimal(max_diff(i, f)[0]), 4) for i in range(3, 31)]),
         'c_max': pd.Series([round(Decimal(max_diff(i, f)[1]), 4) for i in range(3, 31)])}
    df = pd.DataFrame(d)
    print(tabulate(df, headers='keys', tablefmt='psql'))


def statistics(f, n, s):
    function_name = f.__name__+'(x)'
    label_name = s.__name__+'n(x)'
    plt.ylim([-1.5, 4])
    plt.plot(choose_equidistant_nodes(), f_points(choose_equidistant_nodes(), f), color='b', label=function_name)
    plt.plot(choose_equidistant_nodes(), s(n, f), color='orange', label=label_name)
    plt.legend()
    plt.show()

func = f1
error_table(func)
n_values = [3, 10, 20]
for i in range(len(n_values)):
    statistics(func, n_values[i], P)

