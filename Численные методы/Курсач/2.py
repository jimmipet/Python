import numpy as np
import matplotlib.pyplot as plt

# Уравнения движения


def equations(t, y, sigma):
    V, theta, Z, X = y
    dV_dt = -np.sin(theta) - sigma * V**2
    dtheta_dt = (V**2 - np.cos(theta)) / V
    dZ_dt = V * np.sin(theta)
    dX_dt = V * np.cos(theta)
    return [dV_dt, dtheta_dt, dZ_dt, dX_dt]

# Метод Рунге-Кутта 4-го порядка


def rk4_step(t, y, h, sigma):
    k1 = h * np.array(equations(t, y, sigma))
    k2 = h * np.array(equations(t + 0.5*h, y + 0.5*k1, sigma))
    k3 = h * np.array(equations(t + 0.5*h, y + 0.5*k2, sigma))
    k4 = h * np.array(equations(t + h, y + k3, sigma))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


# Начальные условия
theta0 = 60
V0 = 80
Z0 = 10
X0 = 0
sigma = 0.1

# Шаг и количество шагов
h = 0.0001
N = int(20 / h)

# Массивы для хранения значений
t = np.zeros(N+1)
V = np.zeros(N+1)
theta = np.zeros(N+1)
Z = np.zeros(N+1)
X = np.zeros(N+1)

# Начальные значения
t[0] = 0
V[0] = V0
theta[0] = theta0
Z[0] = Z0
X[0] = X0

# Решение ОДУ методом Рунге-Кутта 4-го порядка
for i in range(N):
    y = [V[i], theta[i], Z[i], X[i]]
    y = rk4_step(t[i], y, h, sigma)
    V[i+1], theta[i+1], Z[i+1], X[i+1] = y
    t[i+1] = t[i] + h

# Построение графика
plt.plot(X, Z)
plt.title('Траектория полета планера')
plt.xlabel('x')
plt.ylim(0, 15)
plt.ylabel('z')
plt.show()
