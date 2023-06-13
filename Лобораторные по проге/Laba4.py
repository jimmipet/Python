import numpy as np
from math import pi, cos

# Задаем начальные данные
a = 1
b = pi
T = 1

# Определяем шаги по пространству и времени
h = 0.1
k = 0.01

# Вычисляем количество шагов по пространству и времени
nx = int(b/h) + 1
nt = int(T/k) + 1

# Задаем начальные условия
u0 = np.zeros((nx, nx))
for i in range(nx):
    for j in range(nx):
        x = i*h
        y = j*h
        u0[i, j] = x**3 + cos(y)

# Задаем граничные условия
u0[0, :] = 0
u0[nx-1, :] = 0
u0[:, 0] = 0
u0[:, nx-1] = 0

# Определяем коэффициенты для явной схемы
r = a**2 * k / h**2
s = a**2 * k / h**2

# Вычисляем решение на каждом временном слое
u = np.zeros((nt, nx, nx))
u[0, :, :] = u0
for t in range(1, nt):
    for i in range(1, nx-1):
        for j in range(1, nx-1):
            u[t, i, j] = (1 - 4*r - 4*s)*u[t-1, i, j] + r*(u[t-1, i+1, j] + u[t-1, i-1, j]) + s*(u[t-1, i, j+1] + u[t-1, i, j-1])

# Выводим результаты
number=0
for i in range(1,nx-1):
    for j in range(1,nx-1):
        number=max(number,u[nt-1,i,j])
print(1/number)

