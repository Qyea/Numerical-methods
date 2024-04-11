import math
from tabulate import tabulate

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        integral += f(x)
    integral *= h
    return integral

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(0, n):
        x = a + i * h
        integral += f(x)
        x = a + (i + 0.5) * h
        integral += 4 * f(x)
        x = a + (i + 1) * h
        integral += f(x)
    integral *= h / 6
    return integral

def runge_estimation(approximation, old_approximation, order):
    return abs(approximation - old_approximation) / (2**order - 1)

def f(x):
    return math.log(1 + x) / (3 * x + 2)**2  # Измените функцию f(x) на вашу

a = 0  # Нижний предел интегрирования
b = 1  # Верхний предел интегрирования
epsilon = 1e-7  # Требуемая точность


#exact_integral = 0.071381  # Точное значение интеграла для сравнения
exact_integral = (1/15) * (5 * math.log(5) - 11 * math.log(2))  # Точное значение интеграла для сравнения

n = 1  # Начальное число разбиений
trapezoidal_old = trapezoidal_rule(f, a, b, n)
simpsons_old = simpsons_rule(f, a, b, n)

flag = True
while True:
    
    n *= 2  # Удваиваем число разбиений на каждой итерации
    trapezoidal_new = trapezoidal_rule(f, a, b, n)
    trapezoidal_error = runge_estimation(trapezoidal_new, trapezoidal_old, 2)
    trapezoidal_absolute_error = abs(trapezoidal_new - exact_integral)

    simpsons_new = simpsons_rule(f, a, b, n)
    simpsons_error = runge_estimation(simpsons_new, simpsons_old, 4)
    simpsons_absolute_error = abs(simpsons_new - exact_integral)
    
    if trapezoidal_error < epsilon and simpsons_error < epsilon:
        break

    trapezoidal_old = trapezoidal_new
    simpsons_old = simpsons_new
    
    if (flag == True):
        trapezoidal_error = '-'
        simpsons_error = '-'    
    flag = False
    
    table = [
        ["Квадратурная формула", "Число разбиений", "Шаг", "Приближенное значение", "Оценка погрешности", "Абсолютная погрешность"],
        ["Трапеции", n, (b - a) / n, trapezoidal_new, trapezoidal_error, trapezoidal_absolute_error],
        ["Симпсон", n, (b - a) / n, simpsons_new, simpsons_error, simpsons_absolute_error]
    ]
    table_str = tabulate(table, headers="firstrow", tablefmt="fancy_grid")
    print(table_str)