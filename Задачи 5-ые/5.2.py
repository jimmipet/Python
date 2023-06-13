import numpy as np

n_values = [5, 10, 15]
a_values = [0.2, 0.5, 1]
taus = [0.1, 0.5, 0.9, 1 , 1.5]

for n in n_values:
    for a in a_values:
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 2
        for i in range(1, n):
            A[i, i-1] = -1 + a
        for i in range(n-1):
            A[i, i+1] = -1 - a
        f = np.zeros(n)
        f[0] = 1 - a
        f[n-1] = 1 + a

        for tau in taus:
                x = np.zeros(n)
                num_iterations_relaxation = 0
                max_iterations = 10000
                epsilon = 1e-6
                while num_iterations_relaxation < max_iterations:
                    x_new = np.zeros(n)
                    for i in range(n):
                        sigma = 0
                        for j in range(n):  
                            if j != i:
                                sigma += A[i, j] * x_new[j]
                        x_new[i] = (f[i] - sigma) / A[i, i]
                        x_new[i] = (1 - tau) * x_new[i] +  tau* x[i]  # релаксационный шаг
                    if np.linalg.norm(x - x_new) < epsilon:
                        break
                    x = x_new
                    num_iterations_relaxation += 1
                print(f"n={n}, a={a}, tau={tau}")
                print(f"решение методом релаксации={x}")
                print(f" количество итераций ={num_iterations_relaxation}")

