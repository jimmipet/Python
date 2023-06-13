import math
from scipy.integrate import quad

# определяем функцию


def f(x):
    return math.sqrt(0.7 * x**2 + 2.3) / (3.2 + math.sqrt(0.8 * x + 1.4))


# задаем параметры интегрирования
a = 0.5
b = 1.9

# вычисляем интеграл методом левых прямоугольников
I, _ = quad(f, a, b, method='left')

# выводим результат
print("Интеграл:", I)
