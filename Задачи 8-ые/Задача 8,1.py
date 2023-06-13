# import numpy as np
# import matplotlib.pyplot as plt

# # Определяем функцию
# def f(x):
#     return x**2 - 50*np.sin(x)

# # Определяем отрезок и количество точек для интерполяции
# a, b = 0, 10
# n = 1000

# # Создаем равномерную сетку точек на отрезке [a,b]
# x = np.linspace(a, b, n)

# # Интерполируем функцию вторым порядком
# y = f(x)
# # создаем новый массив и заполняем нулями
# y_interp = np.zeros(n)
# # для каждой точки на отрезке кроме первой и последней вычисляем локальные интерполяционный полином 2 порядка (линейная интерполяция по формуе y0 + (y1 - y0) * ((x_interp - x0) / (x1 - x0)))  исправить
# for i in range(1, n-1):
#     y_interp[i] = y[i] + (y[i+1] - y[i-1]) * (x[i] - x[i-1]) / (x[i+1] - x[i-1])

# # Находим локальные минимумы функции
# minima = []
# for i in range(1, n-1):
#     # если значения функции меньше двух соседних то это минимум
#     if y_interp[i] < y_interp[i-1] and y_interp[i] < y_interp[i+1]:
#         minima.append((x[i], y_interp[i]))

# # Выводим результаты
# print("Первый локальный минимум: x = {:.3f}, f(x) = {:.3f}".format(minima[0][0], minima[0][1]))
# print("Второй локальный минимум: x = {:.3f}, f(x) = {:.3f}".format(minima[1][0], minima[1][1]))

# # Рисуем график функции
# fig, ax = plt.subplots()
# ax.plot(x, y, label="f(x)")
# ax.scatter([minima[0][0], minima[1][0]], [minima[0][1], minima[1][1]], color="red", label="local minima")
# ax.legend()
# ax.set_xlabel("x")
# ax.set_ylabel("f(x)")
# ax.set_title("Graph of f(x) = x^2 - 50*sin(x)")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию


def f(x):
    return x**2 - 50*np.sin(x)


# Определяем отрезок и количество точек для интерполяции
a, b = 0, 10
n = 1000

# Создаем равномерную сетку точек на отрезке [a,b]
x = np.linspace(a, b, n)

# Интерполируем функцию вторым порядком
y = f(x)
# создаем новый массив и заполняем нулями
y_interp = np.zeros(n)
# для каждой точки на отрезке кроме первой и последней вычисляем локальный интерполяционный полином 2 порядка
for i in range(1, n-1):
    # коэффициенты полинома
    a0 = y[i]
    a1 = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    a2 = 2 * (y[i-1] - y[i]) / ((x[i] - x[i-1]) * (x[i+1] - x[i-1])) + \
        2 * (y[i+1] - y[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i-1]))
    # используем формулу квадратичной интерполяции для вычисления значения полинома в точке x[i]
    y_interp[i] = a0 + a1 * (x[i+1] - x[i]) + a2 * (x[i+1] - x[i]) ** 2

# Находим локальные минимумы функции
minima = []
for i in range(1, n-1):
    # если значения функции меньше двух соседних то это минимум
    if y_interp[i] < y_interp[i-1] and y_interp[i] < y_interp[i+1]:
        minima.append((x[i], y_interp[i]))

# Выводим результаты
print("Первый локальный минимум: x = {:.3f}, f(x) = {:.3f}".format(
    minima[0][0], minima[0][1]))
print("Второй локальный минимум: x = {:.3f}, f(x) = {:.3f}".format(
    minima[1][0], minima[1][1]))


fig, ax = plt.subplots()
ax.plot(x, y, label="f(x)")
ax.plot(x, y_interp, label="interpolation")
ax.scatter([minima[0][0], minima[1][0]], [minima[0][1],
           minima[1][1]], color="red", label="local minima")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Graph of f(x) = x^2 - 50*sin(x)")
plt.show()
