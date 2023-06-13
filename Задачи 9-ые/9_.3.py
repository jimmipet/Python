import numpy as np
import matplotlib.pyplot as plt

# Задаем начальные данные
n = 10  # точки
m = 3  # степень полинома
h = 1/n  # расстояние между точками
x = np.arange(0, n+1) * h
y = 1 - np.cos(x)

# Функция для создания матрицы системы уравнений


def create_matrix(x, m):
    n = len(x)
    A = np.zeros((n, m+1))
    for i in range(n):
        A[i] = np.power(x[i], np.arange(m+1))
    return A

# Функция для решения системы уравнений методом наименьших квадратов


def solve_least_squares(A, y):
    return np.linalg.inv(A.T @ A) @ A.T @ y


# Создаем матрицу системы уравнений
A = create_matrix(x, m)

# Решаем систему уравнений методом наименьших квадратов
c = solve_least_squares(A, y)

# Вычисляем аппроксимирующую функцию
p = np.poly1d(c[::-1])

# Строим графики
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, p(x), label='Fitted line')
plt.legend()
plt.show()
