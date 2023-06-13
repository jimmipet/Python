import numpy as np
import scipy
import math

# первая часть задачи
n = 5  # Размер матрицы
A = np.zeros((n, n))

# Заполняем главную диагональ матрицы A
for i in range(n):
    A[i, i] = i+1

# Заполняем первую строку и последний столбец
A[0, -1] = 1
A[-1, 0] = 1

# Заполняем первый столбец и последнюю строку
for i in range(1, n-1):
    A[i, i-1] = 0
    A[i, i+1] = 0

P, L, U = scipy.linalg.lu(A)
print("Матрица A:")
print(A)
print("LU-разложение матрицы A:")
print("P = ")
print(P)
print("L = ")
print(L)
print("U = ")
print(U)

# вторая часть задачи

n = 10  # задайте нужный размер матрицы
h = 1/n

A = np.zeros((n, n))

for i in range(n):
    A[i, i] = 2 + h**2
    if i > 0:
        A[i, i-1] = -1
        A[i-1, i] = -1

A[0, -1] = -1
A[-1, 0] = -1
print("Матрица A(вторая задача):")
print(A)

f = np.zeros(n)

for i in range(n):
    f[i] = (1 + ((4/(h**2)) * (math.sin(math.pi*h)**2))) * \
        math.sin(2*math.pi*(i-1)*h)

print("Столбец значений f(вторая задача):")
print(f)

x = np.linalg.solve(A, f)
print("Столбец ответов x(вторая задача):")
print(x)


y = np.zeros(n)
for i in range(n):
    y[i] = math.sin(2*math.pi*(i-1)*h)

print("Столбец ответов y(вторая задача):")
print(y)

print(np.array_equal(x, y))
