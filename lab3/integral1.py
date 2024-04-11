from scipy.special import roots_legendre
import math
import numpy as np

a = 0  # Нижний предел интегрирования
b = 1  # Верхний предел интегрирования
n = 4  # Число узлов
epsilon = 1e-7  # Требуемая точность

def f(x):
    return math.log(1 + x) / (3 * x + 2)**2  # Измените функцию f(x) на вашу

exact_integral = (1/15) * (5 * math.log(5) - 11 * math.log(2))  # Точное значение интеграла для сравнения (~0.071381)

A0 = (1/72.0)*(18 + math.sqrt(30))
A2 = (1/72.0)*(18 - math.sqrt(30))
A1, A3 = A0, A2

x0 = 1/2.0 - (math.sqrt(525 - 70*math.sqrt(30)))/70.0
x1 = 1/2.0 + (math.sqrt(525 - 70*math.sqrt(30)))/70.0
x2 = 1/2.0 - (math.sqrt(525 + 70*math.sqrt(30)))/70.0
x3 = 1/2.0 + (math.sqrt(525 + 70*math.sqrt(30)))/70.0

nodes = np.array([x0, x1, x2, x3])
weights = np.array([A0, A1, A2, A3])

scaled_nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)

integral = 0
for i in range(n):
    integral += weights[i] * f(scaled_nodes[i])

approximation = (b - a) * integral
exact_value = (1/15) * (5 * math.log(5) - 11 * math.log(2))

print("Приближенное значение интеграла:", approximation)
print("Точное значение интеграла:", exact_value)
print("Абсолютная погрешность:", abs(approximation - exact_value))