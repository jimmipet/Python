import numpy as np
import matplotlib.pyplot as plt

# Определение функций


def f1(t, y1, y2):
    return y1/(2+2*t) - 2*t*y2


def f2(t, y1, y2):
    return y2/(2+2*t) + 2*t*y1


# Начальные условия и параметры метода
a = 0
b = 2
y1_0 = 1
y2_0 = 0

h_list = []
err1_list = []
err2_list = []

for k in range(1, 11):
    h = 0.01*k
    n = int((b-a)/h)
    # Реализация метода Рунге-Кутта 4-го порядка точности
    t = a
    y1 = y1_0
    y2 = y2_0
    for i in range(n):
        k1_1 = h * f1(t, y1, y2)
        k1_2 = h * f2(t, y1, y2)
        k2_1 = h * f1(t + h/2, y1 + k1_1/2, y2 + k1_2/2)
        k2_2 = h * f2(t + h/2, y1 + k1_1/2, y2 + k1_2/2)
        k3_1 = h * f1(t + h/2, y1 + k2_1/2, y2 + k2_2/2)
        k3_2 = h * f2(t + h/2, y1 + k2_1/2, y2 + k2_2/2)
        k4_1 = h * f1(t + h, y1 + k3_1, y2 + k3_2)
        k4_2 = h * f2(t + h, y1 + k3_1, y2 + k3_2)
        y1 += (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        y2 += (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6
        t += h

    # Сравнение с аналитическими решениями
    y1_true = np.cos(t**2) * np.sqrt(1+t)
    y2_true = np.sin(t**2) * np.sqrt(1+t)

    # Оценка погрешности
    err1 = abs(y1_true - y1)
    err2 = abs(y2_true - y2)

    # Вычисление e/h^4
    e_h4_1 = err1 / (h**4)
    e_h4_2 = err2 / (h**4)

    # Сохранение значений погрешности и шага
    h_list.append(h)
    err1_list.append(e_h4_1)
    err2_list.append(e_h4_2)

    max_e_h4_1 = max(err1_list)
    max_e_h4_2 = max(err2_list)
    idx1 = np.argmax(err1_list)
    idx2 = np.argmax(err2_list)
    h_max_e_h4_1 = h_list[idx1]
    h_max_e_h4_2 = h_list[idx2]


plt.plot(h_list, err1_list, label='y1')
plt.plot(h_list, err2_list, label='y2')
plt.ylim(0, 30)
plt.xlabel('Шаг h')
plt.ylabel('Максимальная погрешность, деленная на $h^{-4}$')
plt.title('Зависимость максимальной погрешности, деленной на h^4, от шага h')
plt.legend()
plt.show()
print(
    f"Для y1 максимальная погрешность, деленная на h^4, равна {max_e_h4_1:.6f}, достигается при h = {h_max_e_h4_1:.4f}")
print(
    f"Для y2 максимальная погрешность, деленная на h^4, равна {max_e_h4_2:.6f}, достигается при h = {h_max_e_h4_2:.4f}")
