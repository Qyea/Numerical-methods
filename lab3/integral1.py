from scipy.special import roots_legendre
from math import sqrt, log 
import numpy as np

a = 0  # Нижний предел интегрирования
b = 1  # Верхний предел интегрирования
n = 4  # Число узлов
epsilon = 1e-8  

nodes = [-sqrt((1 / 35) * (15 - 2 * sqrt(30))),
           sqrt((1 / 35) * (15 - 2 * sqrt(30))),
          -sqrt((1 / 35) * (15 + 2 * sqrt(30))),
           sqrt((1 / 35) * (15 + 2 * sqrt(30)))]
weights = [(1 / 36) * (18 + sqrt(30)),
           (1 / 36) * (18 + sqrt(30)),
           (1 / 36) * (18 - sqrt(30)),
           (1 / 36) * (18 - sqrt(30))]

def f(x):
    return (log(1 + x) / ((3 * x + 2)**2))  

def transform_points():
    return [(a + b) / 2 - 0.5 * (b - a) * t for t in nodes]

exact_integral = (1/15) * (5 * log(5) - 11 * log(2))  # Точное значение интеграла для сравнения (~0.071381)


scaled_nodes = transform_points()

integral = 0
for i in range(n):
    integral += weights[i] * f(scaled_nodes[i])

integral*= (b - a) / 2

print("Приближенное значение интеграла:", integral)
print("Точное значение интеграла:", exact_integral)
print("Абсолютная погрешность:", abs(integral - exact_integral))