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
h = 0.01
y1_0 = 1
y2_0 = 0
n = int((b-a)/h)

# Реализация метода Рунге-Кутта 4-го порядка точности
t = a
y1 = y1_0
y2 = y2_0
err1_list = []
err2_list = []
y1_list = []
y2_list = []
h_list = []
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
    y1_list.append(y1)
    y2_list.append(y2)

    # Вывод численных значений функций
    print("Численное решение:")
    print(f"y1({t}) = {y1:.6f}")
    print(f"y2({t}) = {y2:.6f}")

# # Сравнение с аналитическими решениями
    y1_true = np.cos(t**2) * np.sqrt(1+t)
    y2_true = np.sin(t**2) * np.sqrt(1+t)
    print("\nАналитическое решение:")
    print(f"y1({t}) = {y1_true:.10f}")
    print(f"y2({t}) = {y2_true:.10f}")

# Сохранение значений для построения графиков
t_list = np.arange(a, b, h)
y1_true_list = []
y2_true_list = []

for t in t_list:
    y1_true_list.append(np.cos(t**2) * np.sqrt(1+t))
    y2_true_list.append(np.sin(t**2) * np.sqrt(1+t))
print(y1_list)
# Построение графиков
plt.plot(t_list, y1_list, label='Приближенное решение')
plt.plot(t_list, y1_true_list, label='Точное решение')
plt.legend()
plt.xlabel('t')
plt.ylabel('y1')
plt.title('Решение системы уравнений методом Рунге-Кутта')
plt.show()

plt.plot(t_list, y2_list, label='Приближенное решение')
plt.plot(t_list, y2_true_list, label='Точное решение')
plt.legend()
plt.xlabel('t')
plt.ylabel('y2')


plt.title('Решение системы уравнений методом Рунге-Кутта')
plt.show()


# Оценка погрешности
err1 = abs(y1_true - y1)
err2 = abs(y2_true - y2)
print("\nАбсолютная ошибка:")
print(f"Ошибка  в y1: {err1:.10f}")
print(f"Ошибка в y2: {err2:.10f}")
