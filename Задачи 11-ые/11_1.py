import numpy as np
from scipy.integrate import simps

a = 0
b = 1
n = 1000

def f(x): return 3 - 2*x

def K(x, s): return x - s


x = np.linspace(a, b, n)

A = K(x[:, None], x)

b_array = f(x)

y = np.linalg.solve(A, b_array)


integral_value = simps(K(x[:, None], x)*y, x)

print("Приближенное решение:", y)
print("Значение интеграла:", integral_value)


