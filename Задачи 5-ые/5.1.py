import numpy as np

n_values = [5, 10, 15]  # выбираем значения n
a_values = [0.2, 0.5, 1]  # выбираем значения a

for n in n_values:
    for a in a_values:
        num_iterations_seidel = 0
        num_iterations_jacobi = 0
        A = np.zeros((n, n))  # создаем пустую матрицу размерности n x n

        # Заполняем главную диагональ матрицы
        for i in range(n):
            A[i, i] = 2

        # Заполняем диагонали справа и слева от главной диагонали матрицы
        for i in range(1, n):
            A[i, i-1] = -1 + a
        for i in range(n-1):
            A[i, i+1] = -1 - a

        f = np.zeros(n)
        f[0] = 1 - a
        f[1:n-1] = 0
        f[n-1] = 1 + a

        # Решаем систему уравнений методом Зейделя
        x_z = np.zeros(n)  # начальное приближение для метода Зейделя
        num_iterations_seidel = 0
        while True:
            x_z_new = np.zeros(n)
            for i in range(n):
                s1 = np.dot(A[i, :i], x_z_new[:i])
                s2 = np.dot(A[i, i+1:], x_z[i+1:])
                x_z_new[i] = (f[i] - s1 - s2) / A[i, i]
            num_iterations_seidel += 1
            if np.linalg.norm(x_z_new - x_z, np.inf) < 1e-8:
                break
            x_z = x_z_new
        
        # Решаем систему методом Якоби
        x = np.zeros(n)
        num_iterations_jacobi = 0
        while True:
            x_new = np.zeros(n)
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                s2 = np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (f[i] - s1 - s2) / A[i, i]
            num_iterations_jacobi += 1
            if np.linalg.norm(x - x_new) < 1e-8:
                break
            x = x_new

        print(f"n={n}, a={a}")
        print("Решение методом Зейделя:")
        print(x_new)
        print(f"Количество итераций: {num_iterations_seidel}")
        print("Решение методом Якоби:")
        print(x)
        print(f"Количество итераций: {num_iterations_jacobi}")