import matplotlib.pyplot as plt
import numpy as np

# Задание интервала
a = -3
b = 3

# Задание количества узлов
N = 15

# Граничные условия
f_prime_a = 0.077432
f_prime_b= -0.077432

# Задание функции f(x) = sin(cos(x))
def f(x):
    return np.sin(np.cos(x))

# Массив значений функции f(x) в точках 
def f_points(build_points, n=15):
    return [f(build_points[i]) for i in range(n + 1)]

# Выбор узлов (равноотстоящие)
def equidistant_nodes(n=15):
    return np.linspace(a, b, n+1)

# Массив значений hi
def h_value(n=15):
    points = equidistant_nodes()
    h_values = np.zeros(n+1)
    for i in range(1, n+1): 
        h_values[i] = points[i] - points[i - 1]
        
    return h_values

# Метод прогонки
def solve_tridiagonal(a, b, c, f):
    # Прямой ход (прогонка)
    alpha = [-1, b[0] / c[0]]
    beta = [-1, f[0] / c[0]]
    for i in range(1, N):
        alpha.append(b[i] / (c[i] - a[i] * alpha[i]))
    for i in range(1, N+1):
        beta.append((f[i] + a[i] * beta[i]) / (c[i] - a[i] * alpha[i]))

    x = [beta[N+1]]
    # Обратный ход (обратная прогонка)
    for i in range(N-1, -1, -1):
        x.append(alpha[i + 1] * x[-1] + beta[i + 1])

    return x[::-1]

# Вычисление моментов
def find_M():
    points = equidistant_nodes()
    fp = f_points(points)
    h_values = h_value()
    c = [h_values[1] / 3]
    
    for i in range(1, N):
        c.append((h_values[i] + h_values[i+1]) / 3)
    c.append(1)

    a = [-1]
    for i in range(1, N):
        a.append(-1 * h_values[i] / 6)
    a.append(0)

    b = []
    for i in range(1, N):
        b.append(-1 * h_values[i+1] / 6)
    b.append(-1)

    f = [(fp[1] - fp[0]) / h_values[1] - f_prime_a]
    
    for i in range(1, N):
        f.append((fp[i + 1] - fp[i]) / h_values[i+1] - (fp[i] - fp[i - 1]) / h_values[i])
    f.append(f_prime_b)
    M = np.ones(N+1)
    M = solve_tridiagonal(a, b, c, f)
    return M


def get_index(x):
    points = equidistant_nodes()
    for i in range(1, N+1):
        if points[i - 1] <= x <= points[i]:
            return i


def P(h_value, i, x, nodes, M, f):
    hi = h_value
    result =  M[i - 1] * ((nodes[i] - x) ** 3) / (hi*6) + M[i] * ((x - nodes[i-1]) ** 3)  / (hi*6)
    # Ai*(x_i-x)/h_i +Bi*(x-x_i)/h_i
    result += (f[i - 1] - (hi ** 2 / 6) * M[i - 1]) * ((nodes[i] - x) / hi) + (f[i] - (hi ** 2 / 6 )* M[i]) * ((x - nodes[i - 1]) / hi) 
    return result

# Функция вычисления сплайна
def spline():
    build_points = equidistant_nodes(100)
    points = equidistant_nodes()
    M = find_M()
    h_values = h_value()
    f = f_points(points)
    s = []
    for x in build_points:
        i = get_index(x)
        s.append(P(h_values[i], i, x, points, M, f))
    return s

# Вычисление ошибок
def error():
    return np.abs(np.array(f_points(equidistant_nodes(100), 100)) - np.array(spline()))

# Вычисление узлов
x_nodes = equidistant_nodes()
x_interp = equidistant_nodes(100)

# Вычисление значений функции f(x)
f_values = f(x_nodes)
f_values_interp = f(x_interp) # в узлах x*_i

# Вычисление сплайна
spline_values = spline()

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_interp, f_values_interp, label='f(x)')
plt.plot(x_interp, spline_values, color='orange', label='S3(x)')
plt.scatter(x_nodes, f_values, color='red', label='Узлы')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяционный кубический сплайн')
plt.legend()
plt.show()

max_error = (max(error()))
print(f'Max погрешность: {max_error}')

# Построение графика погрешности
plt.figure(figsize=(10, 4))
plt.plot(equidistant_nodes(100), error(), label='Погрешность')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешность интерполирования кубическим сплайном')
plt.legend()
plt.show()
