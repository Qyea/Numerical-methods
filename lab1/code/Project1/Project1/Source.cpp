#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
const double pi = 3.14159265358979323846;

// Функция f1(x) = sin(cos(x))
double f1(double x) {
    return sin(cos(x));
}

// Функция f2(x) = ||x| - 1|
double f2(double x) {
    return abs(abs(x) - 1);
}

// Функция для вычисления разделенных разностей
double divided_difference(std::vector<double>& x, std::vector<double>& y, int start, int end) {
    if (start == end) {
        return y[start];
    }
    else {
        return (divided_difference(x, y, start + 1, end) - divided_difference(x, y, start, end - 1)) / (x[end] - x[start]);
    }
}

// Функция для вычисления значения интерполяционного многочлена в форме Ньютона
double newton_interpolation(std::vector<double>& x, std::vector<double>& y, double point) {
    double result = y[0];
    double term = 1.0;

    for (int i = 1; i < x.size(); i++) {
        term *= (point - x[i - 1]);
        result += term * divided_difference(x, y, 0, i);
    }

    return result;
}

// Функция для создания графика функции и интерполяционного многочлена
void create_plot(std::vector<double>& x, std::vector<double>& y, std::string filename, double (*func)(double)) {
    std::ofstream file(filename);

    for (double i = x[0]; i <= x.back(); i += 0.01) {
        file << i << " " << func(i) << " " << newton_interpolation(x, y, i) << "\n";
    }

    file.close();
}

int main() {
    std::vector<double> x1, y1; // Для сетки равноотстоящих узлов
    std::vector<double> x2, y2; // Для сетки Чебышевских узлов

    double a = 3.0;
    double b = 3.0;
    int n = 30;

    // Генерация узлов равноотстоящей сетки
    for (int i = 0; i <= n; i++) {
        double point = a + i * (b - a) / n;
        x1.push_back(point);
        y1.push_back(f1(point));
        x2.push_back(point);
        y2.push_back(f2(point));
    }

    // Генерация узлов Чебышевской сетки
    for (int i = 0; i <= n; i++) {
        double point = (a + b) / 2.0 + (b - a) / 2.0 * cos((2 * i + 1) * pi / (2 * (n + 1)));
        x2[i] = point;
        y2[i] = f2(point);
    }

    // Создание графиков для f1(x)
    create_plot(x1, y1, "f1_equal_nodes.dat", f1);
    create_plot(x2, y2, "f1_chebyshev_nodes.dat", f1);

    // Создание графиков для f2(x)
    create_plot(x1, y1, "f2_equal_nodes.dat", f2);
    create_plot(x2, y2, "f2_chebyshev_nodes.dat", f2);

    // Создание таблицы для погрешностей интерполяции

    std::ofstream table("errors_table.txt");
    table << "n\tP1,n\tC1,n\tP2,n\tC2,n\n";

    for (int i = 3; i <= 30; i++) {
        std::vector<double> x1_temp(x1.begin(), x1.begin() + i + 1);
        std::vector<double> y1_temp(y1.begin(), y1.begin() + i + 1);
        std::vector<double> x2_temp(x2.begin(), x2.begin() + i + 1);
        std::vector<double> y2_temp(y2.begin(), y2.begin() + i + 1);

        double max_error_P1n = 0.0;
        double max_error_C1n = 0.0;
        double max_error_P2n = 0.0;
        double max_error_C2n = 0.0;

        // Вычисление погрешностей для P1,n и C1,n
        for (double xi = a; xi <= b; xi += 0.01) {
            double error_P1n = abs(newton_interpolation(x1_temp, y1_temp, xi) - f1(xi));
            double error_C1n = abs(newton_interpolation(x2_temp, y2_temp, xi) - f1(xi));
            if (error_P1n > max_error_P1n) {
                max_error_P1n = error_P1n;
            }
            if (error_C1n > max_error_C1n) {
                max_error_C1n = error_C1n;
            }
        }

        // Вычисление погрешностей для P2,n и C2,n
        for (double xi = a; xi <= b; xi += 0.01) {
            double error_P2n = abs(newton_interpolation(x1_temp, y1_temp, xi) - f2(xi));
            double error_C2n = abs(newton_interpolation(x2_temp, y2_temp, xi) - f2(xi));
            if (error_P2n > max_error_P2n) {
                max_error_P2n = error_P2n;
            }
            if (error_C2n > max_error_C2n) {
                max_error_C2n = error_C2n;
            }
        }

        table << i << "\t" << max_error_P1n << "\t" << max_error_C1n << "\t" << max_error_P2n << "\t" << max_error_C2n << "\n";
    }

    table.close();

    return 0;
}