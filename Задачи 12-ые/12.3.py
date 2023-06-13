import numpy as np


def f(t, u, v):
    return -np.sin(u)


def schroeder_method(tau, T, u0, v0):
    num_points = int(T / tau)  # Количество точек с шагом tau
    t_values = np.linspace(0, T, num=num_points+1)
    u_values = np.zeros(num_points+1)
    v_values = np.zeros(num_points+1)
    u_values[0] = u0
    v_values[0] = v0

    for i in range(num_points):
        t = t_values[i]
        u = u_values[i]
        v = v_values[i]

        # Вычисление k1
        k1u = tau * v
        k1v = tau * f(t, u, v)

        # Вычисление k2
        k2u = tau * (v + 0.5 * k1v)
        k2v = tau * f(t + 0.5 * tau, u + 0.5 * k1u, v + 0.5 * k1v)

        # Вычисление k3
        k3u = tau * (v + 2 * k2v - k1v)
        k3v = tau * f(t + tau, u + 2 * k2u - k1u, v + 2 * k2v - k1v)

        # Вычисление новых значений u и v
        u_values[i+1] = u + (1/6) * (k1u + 4 * k2u + k3u)
        v_values[i+1] = v + (1/6) * (k1v + 4 * k2v + k3v)

    return t_values, u_values


# Заданные значения
T = 4 * np.pi
u0 = 1
v0 = 0
tau = 0.01

# Решение методом Штермера
t_values, u_values = schroeder_method(tau, T, u0, v0)

# Вывод результатов
for t, u in zip(t_values, u_values):
    print(f"t = {t:.6f}, u = {u:.6f}")
