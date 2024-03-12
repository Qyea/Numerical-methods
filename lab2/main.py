import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# Задание интервала
a = -3
b = 3

# Задание количества узлов
N = 15

h = (b-a)/N

# Задание функции f(x) = sin(cos(x))
def f(x):
    return np.sin(np.cos(x))

def df(x):
    return (-np.cos(np.cos(x))*np.sin(x))

def solve_tridiagonal(a, b, c, f):
    n = len(f)
    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -b[0] / c[0]
    beta[0] = f[0] / c[0]

    for i in range(1, n-1):
        alpha[i] = -b[i] / (c[i] + a[i-1] * alpha[i - 1])
    for i in range(1, n):
        beta[i] = (f[i] - a[i-1] * beta[i - 1]) / (c[i] + a[i-1] * alpha[i - 1])

    # Обратный ход (обратная прогонка)
    x = [0.0] * n
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x
    
def compute_S3(x, M, f):
    S3_x = [lambda i: lambda x: ( M[i - 1] * ((x[i] - x) ** 3) / (6 * h) + M[i] * ((x - x[i-1]) ** 3) / (6 * h) + (f[i - 1] - (h * h) * M[i - 1] / 6) * (x[i] - x) / h + (f[i] - (h * h) * M[i] / 6) * (x - x[i-1]) / h) for i in range(1, N + 1)]
    return S3_x

def createS3(i, x_nodes, M, f):
    def s(x): 
        return (M[i - 1] * ((x_nodes[i] - x) ** 3) / (6 * h) + M[i] * ((x - x_nodes[i-1]) ** 3) / (6 * h) + (f[i - 1] - (h * h) * M[i - 1] / 6) * (x_nodes[i] - x) / h + (f[i] - (h * h) * M[i] / 6) * (x - x_nodes[i-1]) / h)
    return s


# Вычисление узлов и значений функции в узлах
x_nodes = np.linspace(a, b, N+1)
f_values = f(x_nodes)


f_vector = []
f_vector.append((f_values[1] - f_values[0]) / h - df(a)) 
for i in range(1,N): 
    f_vector.append((f_values[i+1] - f_values[i])/h - (f_values[i]-f_values[i-1])/h)
f_vector.append(df(b) - (f_values[N] - f_values[N-1]) / h)

upper_diagonal = []
middle_diagonal = []
lower_diagonal = []

middle_diagonal.append(h / 3)
upper_diagonal.append(h / 6) 

for i in range(1, N):
    upper_diagonal.append(h / 6)
    middle_diagonal.append((2*h) / 3)
    lower_diagonal.append(h / 6)

lower_diagonal.append(h / 6)
middle_diagonal.append(h / 3)


M_values = solve_tridiagonal(lower_diagonal, upper_diagonal, middle_diagonal, f_vector)

S3_x = []

for i in range(1, N+1):
    S3_x.append(createS3(i, x_nodes, M_values, f_values))

def S(x):
    i  = int((x - a)/h)
    if i == N:
        i = (N-1)
    return S3_x[i](x)

def error():
    return np.abs(np.array(f_values_interp) - np.array(S3x))

x_interp = np.linspace(a, b, 101)
f_values_interp = f(x_interp)

S3x = [S(x) for x in x_interp]

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_interp, f_values_interp, label='f(x)')
plt.plot(x_interp, S3x, color='orange', label='S3(x)')
plt.scatter(x_nodes, f_values, color='red', label='Узлы')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяционный кубический сплайн')
plt.legend()
plt.show()

max_error = np.max(np.abs(S3x - f_values_interp))
print(max_error)

plt.figure(figsize=(10, 4))
plt.plot(x_interp, error(), label='Погрешность')
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Погрешность интерполирования кубическим сплайном')
plt.legend()
plt.show()