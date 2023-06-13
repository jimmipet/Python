import numpy as np
import scipy


# Создаем матрицу Лемера
n = 5
A = np.zeros((n, n))
for i in range(1, n+1):
    for j in range(1, n+1):
        A[i-1, j-1] = min(i, j)/max(i, j)
print("A:\n", A)
# Вычисляем LU разложение с выбором главного элемента по строке
P, L, U = scipy.linalg.lu(A)


# ответ проверил на калькуляторе
X = np.linalg.inv(A)
X = np.around(X, decimals=3)  # округляем до 3 знаков после запятой
X[np.abs(X) < 1e-10] = 0  # заменяем очень маленькие значения на 0
print(A.dot(X))
print("A^(-1):\n", X)
