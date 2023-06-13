import numpy as np
from scipy.optimize import minimize_scalar

# Задаем функцию f(x,y)
def f(x, y):
    a = 1
    b = -1.4
    c = 0.01
    d = 0.11
    return a*x + b*y + np.exp(c*x**2 + d*y**2)

# Вычисляем градиент функции f(x,y)
def gradient(x, y):
    a = 1
    b = -1.4
    c = 0.01
    d = 0.11
    grad_x = a + 2*c*x*np.exp(c*x**2 + d*y**2)
    grad_y = b + 2*d*y*np.exp(c*x**2 + d*y**2)
    return grad_x, grad_y

# Функция градиентного спуска
def gradient_descent(x_init, y_init, alpha, epsilon):
    x = x_init
    y = y_init
    while True:
        grad_x, grad_y = gradient(x, y)
        new_x = x - alpha*grad_x
        new_y = y - alpha*grad_y
        if abs(f(new_x, new_y) - f(x, y)) < epsilon:
            break
        else:
            x = new_x
            y = new_y
    return new_x, new_y

# Проверяем свойство выпуклости функции графически
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Вычисляем минимум функции с помощью градиентного спуска
epsilon = 0.000001
alpha = 0.1
x_init = 0
y_init = 0
x_min, y_min = gradient_descent(x_init, y_init, alpha, epsilon)

print("Минимум достигается в точке (", x_min, ", ", y_min, ")")
print("Значение функции в этой точке: ", f(x_min, y_min))

# Проверяем результат с помощью методов оптимизации из библиотеки SciPy
def g(y):
    return f(x_min, y)
res = minimize_scalar(g)
print("Минимум функции f(x,y) равен ", res.fun, " достигается в точке y = ", res.x)
