import numpy as np

def runge_kutta_fehlberg(f, t0, u0, tf, h):
    """
    Реализация метода Рунге-Кутты-Фельберга для решения дифференциальных уравнений.

    Параметры:
    f: функция правой части дифференциального уравнения du/dt = f(t, u)
    t0: начальное время
    u0: начальное значение решения
    tf: конечное время
    h: шаг интегрирования

    Возвращает:
    t_values: список значений времени
    u_values: список значений решения
    """
    
    t_values = [t0]
    u_values = [u0]
    
    while t0 < tf:
        # Вычисляем коэффициенты метода Рунге-Кутты-Фельберга
        k1 = h * f(t0, u0)
        k2 = h * f(t0 + h/4, u0 + k1/4)
        k3 = h * f(t0 + 3*h/8, u0 + 3*k1/32 + 9*k2/32)
        k4 = h * f(t0 + 12*h/13, u0 + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(t0 + h, u0 + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(t0 + h/2, u0 - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
        
        # Вычисляем приближенные значения решения
        u1 = u0 + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        u2 = u0 + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        
        # Оценка погрешности метода
        tau = h * np.abs(u2 - u1)
        eps = 0.001  # заданная точность
        
        if tau <= eps:
            # Приближенное значение достаточно точно
            t0 += h
            u0 = u1
            t_values.append(t0)
            u_values.append(u0)
        
        # Адаптация шага интегрирования
        h = h * min(max((eps / tau)**0.25, 0.1), 4.0)
    
    return t_values, u_values

def f(t, u):
    """
    Функция правой части дифференциального уравнения du/dt = f(t, u)
    """
    return 1 + u**2

# Заданные значения
t0 = 0
u0 = 0
tf = 1
h = 0.1

t_values, u_values = runge_kutta_fehlberg(f, t0, u0, tf, h)

for t, u in zip(t_values, u_values):
    print(f"t = {t:.2f}, u = {u:.6f}")
