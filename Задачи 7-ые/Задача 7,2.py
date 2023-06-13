import math
import matplotlib.pyplot as plt

# Определяем функцию и её производную
def f(x): return x**2 - 10*math.sin(x)
def df(x): return 2*x - 10*math.cos(x)

# Определяем функцию метода Ньютона
def newton(f, df, x0, eps):
    while abs(f(x0)) > eps:
        x0 = x0 - f(x0) / df(x0)
    return x0


# Создаем список положительных корней
roots = []
for x0 in range(1):
    root = newton(f, df, x0, 1e-6)
    if root >= 0:
        roots.append(root)


print("Положительные корни уравнения x^2 - 10*sin(x) = 0:", roots)


# Рисуем график функции
x_values = [x/100 for x in range(-2000, 2000)]
y_values = [f(x) for x in x_values]

plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции y = x^2 - 10*sin(x)')

# Помечаем корни на графике
for root in roots:
    plt.scatter(root, f(root), color='red')

plt.show()
