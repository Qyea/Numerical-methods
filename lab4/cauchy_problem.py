import matplotlib.pyplot as plt
import numpy as np

a = 1
b = 2
h_v = 0.2
y0 = np.exp(1)
p = 2

def f(x, u):
    return (u * np.log(u)) / x

# Точное значение в узле
def exact_solution(x):
    return np.exp(x)

# Получение узлов
def get_x(h):
    N = int((b - a) / h)
    return [a + i * h for i in range(N + 1)]

# Получаем численные решения
def get_exact(h=0.1):
    x_exact = get_x(h)
    return [exact_solution(x) for x in x_exact]

def runge(y_h, y_2h, h=0.2):
    N = int((b - a) / h)
    i, j = 0, 0
    max_diff = 0
    while i != N + 1:
        max_diff = max(max_diff, abs(y_h[i] - y_2h[j]))
        i += 1
        j += 2
    return max_diff / (2 ** p - 1)

def max_norm(u, y):
    max_diff = max(abs(u - y))
    return max_diff

def implicit_trapezoidal_method(y0, h=0.2):
    N = int((b - a) / h)
    x = np.linspace(a, b, N+1)
    y = np.zeros(N+1)
    y[0] = y0

    for i in range(1, N+1):
        y_pred = y[i-1]
        for _ in range(N+1):
            y_pred = y[i-1] + h/2 * (f(x[i-1], y[i-1]) + f(x[i], y_pred))

        y[i] = y_pred

    return x, y

def explicit_trapezoidal_method(y0, h=0.2):
    N = int((b - a) / h)  
    x = np.linspace(a, b, N+1)  
    y = np.zeros(N+1)  
    y[0] = y0 
    
    for i in range(1, N+1):
        y[i] = y[i-1] + h * (f(x[i-1], y[i-1]) + f(x[i], y[i-1] + h * f(x[i-1], y[i-1]))) / 2

    return x, y

def adams_predictor_corrector_method(y0, explicit_value, h=0.2):
    N = int((b - a) / h) 
    x = np.linspace(a, b, N+1) 
    y = np.zeros(N+1) 
    y[0] = y0  

    # Используем явный метод Эйлера для первого шага
    y[1] = explicit_value

    for i in range(2, N+1):
        # Предикторный шаг
        y_pred = y[i-1] + (h/2) * (3 * f(x[i-1], y[i-1]) - f(x[i-2], y[i-2]))
        # Корректорный шаг
        y[i] = y[i-1] + (h/2) * (f(x[i], y_pred) + f(x[i-1], y[i-1]))

    return x, y

n = int((b - a) / h_v)
x_exact = [a + i * h_v for i in range(n + 1)]
u_exact = exact_solution(x_exact)

x_explicit, y_explicit = explicit_trapezoidal_method(y0)
x_implicit, y_implicit = implicit_trapezoidal_method(y0)
x_adams, y_adams = adams_predictor_corrector_method(y0, y_explicit[1])

print("x\tExact\tExplicit\tImplicit\tAdams")
for i in range(len(x_exact)):
    print("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
            x_exact[i], u_exact[i], y_implicit[i],
            y_explicit[i], y_adams[i]))

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_exact, u_exact, label='Exact Solution')
plt.plot(x_explicit, y_explicit, label='Explicit Trapezoidal Method')
plt.plot(x_implicit, y_implicit, label='Implicit Trapezoidal Method')
plt.plot(x_adams, y_adams, label='Adams Method')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Numerical Solutions')
plt.legend()
plt.grid(True)
plt.show()

h_2 = h_v / 2
n = int((b - a) / h_2)
x_exact = [a + i * h_2 for i in range(n + 1)]
u_exact = exact_solution(x_exact)

x_2h_explicit, y_2h_explicit = explicit_trapezoidal_method(y0, h_2)
error_explicit = runge(y_explicit, y_2h_explicit)

x_2h_implicit, y_2h_implicit = implicit_trapezoidal_method(y0, h_2)
error_implicit = runge(y_implicit, y_2h_implicit)

x_2h_adams, y_2h_adams = adams_predictor_corrector_method(y0, y_2h_explicit[1], h_2)
error_adams= runge(y_adams, y_2h_adams)


print("x\tExact\tExplicit\tImplicit\tAdams")
for i in range(len(x_exact)):
    print("{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
            x_exact[i], u_exact[i], y_2h_implicit[i],
            y_2h_explicit[i], y_2h_implicit[i]))

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x_exact, u_exact, label='Exact Solution')
plt.plot(x_2h_explicit, y_2h_explicit, label='Explicit Trapezoidal Method')
plt.plot(x_2h_implicit, y_2h_implicit, label='Implicit Trapezoidal Method')
plt.plot(x_2h_adams, y_2h_adams, label='Adams Method')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Numerical Solutions')
plt.legend()
plt.grid(True)
plt.show()

errors_norm_explicit = max(abs(ui - yi) for ui, yi in zip(u_exact, y_2h_explicit))
errors_norm_implicit = max(abs(ui - yi) for ui, yi in zip(u_exact, y_2h_implicit))
errors_norm_adams = max(abs(ui - yi) for ui, yi in zip(u_exact, y_2h_adams))
    
print("\nErrors:")
print("Explicit:\t", "norm:", errors_norm_explicit, "runge:", error_explicit)
print("Implicit:\t", "norm:", errors_norm_implicit, "runge:", error_implicit)
print("Adams:\t", "norm:", errors_norm_adams)

