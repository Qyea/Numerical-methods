import numpy as np
from scipy.optimize import curve_fit

def polynomial_func(x, a, c):
    return a * x**2 + x + c

# Задаем данные
x_data = np.linspace(-1, 1, 100)
y_data = np.random.normal(0, 0.1, size=100)  # Добавляем шум к данным

# Выполняем аппроксимацию с использованием метода наименьших квадратов
popt, pcov = curve_fit(polynomial_func, x_data, y_data)

# Получаем коэффициенты многочлена
a_opt, b_opt, c_opt = popt

print(f"Наименьше уклоняющийся от нуля многочлен: {a_opt}x^2 + {b_opt}x + {c_opt}")