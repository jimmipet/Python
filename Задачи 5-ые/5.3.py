import numpy as np
import time
from scipy.sparse.linalg import cg


# Функция для вычисления правой части системы
def right_part(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1/(i+j+1)
    f = np.sum(A, axis=1)
    return f

# Функция для явного итерационного метода минимальных поправок
def min_correction(A, f, eps=1e-6):
    n = len(f)
    x = np.zeros(n)
    r = f - np.dot(A, x)  # вычисление начальной невязки
    s = r.copy()
    num_iter = 0
    while np.linalg.norm(r) > eps:  # условие
        # вычисление параметра alpha
        alpha = np.dot(r, r) / np.dot(s, np.dot(A, s))
        x = x + alpha * s  # коррекция решения
        r_new = r - alpha * np.dot(A, s)  # вычисление новой невязки
        beta = np.dot(r_new, r_new) / np.dot(r, r)  # вычисление параметра beta
        s = r_new + beta * s  # вычисление нового направления
        r = r_new  # обновление значения невязки
        num_iter += 1
    return x, num_iter


n_values = [10, 20, 50, 100, 200, 500, 1000]  # значения n для тестирования

print("Метод сопряженных градиентов:")

for n in n_values:
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
    f = np.sum(A, axis=1)
    x_exact = np.ones(n)
    start_time = time.time()
    x, num_iter = cg(A, f)
    end_time = time.time()
    print(
        f"n = {n}, число итераций: {num_iter}, время: {end_time - start_time:.6f} секунд")

print("\nЯвный метод минимальных поправок:")

for n in n_values:
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
    f = np.sum(A, axis=1)
    x_exact = np.ones(n)
    start_time = time.time()
    x, num_iter = min_correction(A, f)
    end_time = time.time()
    print(
        f"n = {n}, число итераций: {num_iter}, время: {end_time - start_time:.6f} секунд")
