import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return math.pow(x, 2) + 5 * (math.exp(-x / 4))


def df(x):
    return (8 * x * math.exp(x / 4) - 5) / (4 * math.exp(x / 4))


if __name__ == "__main__":
    e = 0.000001
    a = 0
    b = 1
    N = 4
    y1 = f(a)
    y2 = f(b)
    z1 = df(a)
    z2 = df(b)
    while b - a > 2 * e:
        s = ((z2 * b - z1 * a) - (y2 - y1)) / (z2 - z1)
        y = f(s)
        z = df(s)
        N += 2
        if z == 0:
            x = s
            break
        if z > 0:
            b = s
            y2 = y
            z2 = z
        else:
            a = s
            y1 = y
            z1 = z
    if z != 0:
        x = (a + b) / 2
        y = f(x)

    # Создаем массив точек для построения графика функции
    x_vals = np.linspace(0, 1, 100)
    y_vals = [f(x) for x in x_vals]

    # Строим график функции
    plt.plot(x_vals, y_vals, label="f(x)")
    plt.scatter(x, y, color="red", label="Root")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graph of f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Metod kasatelnyh\n")
    print(f"x = {x:.5f}")
    print(f"y = {y:.5f}\n")
    print(f"Kol-vo experimentov: {N}")
