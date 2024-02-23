import numpy as np
import matplotlib.pyplot as plt
import math
    
# Определение функции f1(x) = sin(cos(x))
def f1(x):
    return np.sin(np.cos(x))

# Определение функции f2(x) = ||x| - 1|
def f2(x):
    return np.abs(np.abs(x) - 1)

# Выбор узлов (равноотстоящие)
def choose_equidistant_nodes(n, a, b):
    # step = (b-a)/n => for(i){nodesArray+= a+step*i;}
    return np.linspace(a, b, n + 1) 

# (чебышёвские)
def choose_chebyshev_nodes(n, a, b):
    k = np.arange(n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    return nodes

# Построение интерполяционных многочленов в форме Ньютона
def newton_interpolation(x, y):
    n = len(x) - 1
    coefficients = y.copy()

    for i in range(1, n + 1):
        for j in range(0, n + 1 - i):
            coefficients[j+i] = (coefficients[j+i] - coefficients[j+i-1]) / (x[j+i] - x[j])

    return coefficients


def draw_graphics(n_values, f):
    for i in range(len(n_values)):
        # Равноотстоящие узлы для интерполяции f(x)
        equidistant_nodes = choose_equidistant_nodes(n_values[i], -3, 3)
        equidistant_coefficients = newton_interpolation(equidistant_nodes, f(equidistant_nodes))
        equidistant_interpolated = np.polyval(equidistant_coefficients[::-1], x)

        # Чебышевские узлы для интерполяции f(x)
        chebyshev_nodes = choose_chebyshev_nodes(n_values[i], -3, 3)
        chebyshev_coefficients = newton_interpolation(chebyshev_nodes, f(chebyshev_nodes))
        chebyshev_interpolated = np.polyval(chebyshev_coefficients[::-1], x)
        
        function_name = f.__name__+'(x)'
        
        plt.figure(figsize=(12, 4))
        plt.ylim([-1,3])
        plt.subplot(121)
        plt.plot(x, f(x), label= function_name)
        plt.plot(x, equidistant_interpolated, label='P1,n (equidistant)')
        plt.title('Interpolation of '+ function_name+ ' - Equidistant Nodes')
        plt.legend()

        plt.subplot(122)
        plt.plot(x, f(x), label=function_name)
        plt.plot(x, chebyshev_interpolated, label='C1,n (Chebyshev)')
        plt.title('Interpolation of '+ function_name+ ' - Chebyshev Nodes')
        plt.legend()

def print_statistics(max_errors_equidistant, max_errors_chebyshev, n_values, function_name):
    file_name = function_name + ".txt"
    print("n    Max Error (Equidistant)    Max Error (Chebyshev)")
    for i in range(len(n_values)):
        print(f"{n_values[i]:2d}   {max_errors_equidistant[i]:.6f}     {max_errors_chebyshev[i]:.6f}")
        
    with open(file_name, "w") as file:
        print("n    Max Error (Equidistant)    Max Error (Chebyshev)", file=file)
        for i in range(len(n_values)):
            print(f"{n_values[i]:2d}   {max_errors_equidistant[i]:.6f}     {max_errors_chebyshev[i]:.6f}", file=file)
        

def calculate_statistics(f, n_values):
    max_errors_equidistant = []
    max_errors_chebyshev = []
    for n in n_values:
        # Равноотстоящие узлы для интерполяции f(x)
        equidistant_nodes = choose_equidistant_nodes(n, -3, 3)
        equidistant_coefficients = newton_interpolation(equidistant_nodes, f(equidistant_nodes))
        equidistant_interpolated = np.polyval(equidistant_coefficients[::-1], x)
        max_error_equidistant = np.max(np.abs(equidistant_interpolated - f(x)))
        max_errors_equidistant.append(max_error_equidistant*((3-(-3))**(n+1))/(math.factorial(n+1)*(2**(2*n + 1))))

        # Чебышевские узлы для интерполяции f(x)
        chebyshev_nodes = choose_chebyshev_nodes(n, -3, 3)
        chebyshev_coefficients = newton_interpolation(chebyshev_nodes, f(chebyshev_nodes))
        chebyshev_interpolated = np.polyval(chebyshev_coefficients[::-1], x)
        max_error_chebyshev = np.max(np.abs(chebyshev_interpolated - f(x)))
        max_errors_chebyshev.append(max_error_chebyshev*((3-(-3))**(n+1))/(math.factorial(n+1)*(2**(2*n + 1))))
        function_name = f.__name__
    print_statistics(max_errors_equidistant, max_errors_chebyshev, n_values, function_name)    
    
# Генерация точек на оси x для построения графиков
x = np.linspace(-3, 3, 100)

n_values = [3, 10, 20]  # Значения n для интерполяции

# Интерполяция функции f1(x)
draw_graphics(n_values, f1)

# Интерполяция функции f2(x)
draw_graphics(n_values, f2)
# Таблица погрешностей

plt.show()

# Создание таблицы погрешностей
n_values = range(3, 31)  # Значения n для интерполяции

calculate_statistics(f1, n_values)
calculate_statistics(f2, n_values)