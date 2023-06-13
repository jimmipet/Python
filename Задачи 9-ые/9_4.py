import numpy as np
import matplotlib.pyplot as plt

# Задаем начальные данные
n = 10  # количество точек
h = 1/n  # шаг
x = np.arange(0, n+1) * h  # сетка по x
y = 1 - np.cos(x)  # сеточная функция

# Определяем функцию xi(x)


def xi(x, a, b):
    """
    Функция xi(x) для аппроксимации
    """
    return a * np.exp(b*x)


# Создаем матрицу системы уравнений
A = np.zeros((n+1, 2))
A[:, 0] = np.exp(x)
A[:, 1] = x * np.exp(x)

# Решаем систему уравнений методом наименьших квадратов
c, resid, rank, sigma = np.linalg.lstsq(A, y, rcond=None)

# Вычисляем аппроксимирующую функцию
a, b = c
def p(x): return xi(x, a, b)


# Строим графики
plt.plot(x, y, 'ro', label='Original data')  # сеточная функция
plt.plot(x, p(x), label='Fitted line')  # аппроксимирующая функция
plt.legend()
plt.show()
