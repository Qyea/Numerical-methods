import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Задание интервала
a = -3
b = 3

# Задание функции f(x) = sin(cos(x))
def f(x):
    return np.sin(np.cos(x))

# Задание количества узлов
N = 15

# Вычисление узлов и значений функции в узлах
x_nodes = np.linspace(a, b, N+1)
f_values = f(x_nodes)

# Вычисление производных на концах интервала
f_prime_a = np.gradient(f_values)[0]
f_prime_b = np.gradient(f_values)[-1]

# Построение интерполяционного кубического сплайна
spline = CubicSpline(x_nodes, f_values, bc_type=((1, f_prime_a), (1, f_prime_b)))

# Вычисление значений сплайна в узлах x*_i
x_interp = np.linspace(a, b, 101)
spline_values = spline(x_interp)

# Вычисление значений функции f(x) в узлах x*_i
f_values_interp = f(x_interp)

# Вычисление максимальной погрешности
max_error = np.max(np.abs(spline_values - f_values_interp))

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_interp, f_values_interp, label='f(x)')
plt.plot(x_interp, spline_values, label='S3(x)')
plt.scatter(x_nodes, f_values, color='red', label='Узлы')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяционный кубический сплайн')
plt.legend()

# Построение графика погрешности
plt.figure(figsize=(10, 4))
plt.plot(x_interp, np.abs(spline_values - f_values_interp), label='Погрешность')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешность интерполирования кубическим сплайном')
plt.legend()

plt.show()

# Вывод максимальной погрешности
print(f'Max погрешность: {max_error}')