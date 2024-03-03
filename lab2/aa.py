import numpy as np
import matplotlib.pyplot as pltimport 
from decimal import Decimal

# Задание интервала
a = -3
b = 3

# Задание количества узлов
N = 15

# Задание функции f(x) = sin(cos(x))
def f(x):
    return np.sin(np.cos(x))

def solve_tridiagonal(a1, b2, c, f, x, n):
    # Прямой ход (прогонка)
    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -b2[0] / c[0]
    beta[0] = f[0] / c[0]

    for i in range(1, n):
        denominator = 1.0 / (c[i] + a1[i] * alpha[i - 1])  # знаменатель для alpha и beta
        alpha[i] = -b2[i] * denominator
        beta[i] = (f[i] - a1[i] * beta[i - 1]) * denominator

    # Обратный ход (обратная прогонка)
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]


def compute_S3(x, h_values, M, f_values):
    S3_x = []
    for i in range(1, len(x)):
        term1 = M[i - 1] * ((x[i] - x) ** 3) / (6 * h_values[i])
        term2 = M[i] * ((x - x[i-1]) ** 3) / (6 * h_values[i])
        term3 = (f_values[i - 1] - (h_values[i] ** 2) * M[i - 1] / 6) * (x[i] - x) / h_values[i]
        term4 = (f_values[i] - (h_values[i] ** 2) * M[i] / 6) * (x - x[i-1]) / h_values[i]
        S3_x.append(term1 + term2 + term3 + term4)
    return S3_x

# Вычисление узлов и значений функции в узлах
x_nodes = np.linspace(a, b, N+1)
f_values = f(x_nodes)

# Вычисление производных на концах интервала
f_prime_a = np.gradient(f_values)[0]
f_prime_b = np.gradient(f_values)[-1]

h_values = np.zeros(N)
M_values = np.zeros(N)

for i in range(1, N): 
    h_values[i] = Decimal((b - a) / i)
print(h_values)

f_vector = np.zeros(N)
f_vector[0] = (f_values[1] - f_values[0]) / h_values[1] - f_prime_a
f_vector[-1] = f_prime_b - (f_values[-1] - f_values[-2]) / h_values[-1]

for i in range(1,N-1): 
    f_vector[i] = (f_values[i+1] - f_values[i])/h_values[i+1] - (f_values[i]-f_values[i-1])/h_values[i]


upper_diagonal = np.zeros(N)
middle_diagonal = np.zeros(N)
lower_diagonal = np.zeros(N)

for i in range(1, N-1):
    upper_diagonal[i] = h_values[i+1] / 6
    middle_diagonal[i] = (h_values[i] + h_values[i+1]) / 3
    lower_diagonal[i] = h_values[i] / 6

lower_diagonal[N-1] = h_values[-1] / 6
middle_diagonal[N-1] = h_values[-1] / 3
middle_diagonal[0] = h_values[1] / 3
upper_diagonal[0] = h_values[1] / 6

M_values = np.zeros(N)
print(f_values)
solve_tridiagonal(lower_diagonal, middle_diagonal, upper_diagonal, f_vector, M_values, N)
print(M_values)
# Вычисление S3(x) для нескольких значений x
x_values = np.linspace(a, b, 101)  # Пример значений x

f_values_interp = f(x_values)


S3_x = np.zeros(N)
S3_x[0] = f_prime_a
S3_x_add = compute_S3(x_values, h_values, M_values, f_values)
S3_x += S3_x_add
S3_x[N-1] = f_prime_b
print(f"S3({x_values}) =", S3_x)

max_error = np.max(np.abs(S3_x - f_values_interp))



# Вычисление значений функции f(x) в узлах x*_i
"""f_values_interp = f(x_interp)

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
plt.show()

# Вывод максимальной погрешности
print(f'Max погрешность: {max_error}')"""