import numpy as np
import math
import scipy



# Инициализация размерности матрицы
n = 5
# Инициализация матрицы A с помощью вложенных циклов
A = np.zeros((n, n))
for i in range(1, n+1):
    for j in range(1, n+1):
        A[i-1][j-1] = math.factorial(i+j-2) // (math.factorial(i-1) * math.factorial(j-1))

# Вызов функции LU-разложения с выбором главного элемента по строке
P, L, U =scipy.linalg.lu(A)
# Вывод результатов на экран
print("A:\n", A)
print("P:\n", P)
print("L:\n", L)
print("U:\n", U)
print("LU:\n", L @ U)
