import numpy as np
import scipy
# Создание матрицы A
n = 5
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i][j] = min(i+1, j+1)

# Выполнение LU-разложения матрицы A
P, L, U = scipy.linalg.lu(A)

# Вычисление числа обусловленности матрицы A
cond_A = np.linalg.cond(A, "fro")
cond_B= np.linalg.cond(A, np.inf)

# Нахождение обратной матрицы
A_inv = np.linalg.inv(A)

# Печать результатов
print("Матрица A:")
print(A)
print("LU-разложение матрицы A:")
print("P = ")
print(P)
print("L = ")
print(L)
print("U = ")
print(U)
print("Число обусловленности матрицы A:")
print(cond_A)
print(cond_B)
print("Обратная матрица к матрице A:")
print(A_inv)

print(A.dot(A_inv))
