import numpy as np
from numpy import inf
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy import optimize
def f(x):
    return  x**2 - 50*np.sin(x)

#8.1Напишите программу для нахоэюдения минимума функции одной переменной f(x )
# на интервале [а, Ь]

def f1(x):

     return 2*x-50*np.cos(x)
def f2(x):

        return 2 +50*np.sin(x)



def quadratic_approx(x, x0, f, f1, f2):
    return f(x0)+f1(x0)*(x-x0)+(f2(x0))*((x-x0)**2)/2

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt



def newton(x0, fprime, fsecond, maxiter=100, eps=0.001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if (xnew-x)<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x

x = np.linspace(-10, 10)
y=f(x)
plt.plot( x , y, label= '$( x**2 - 50*sin(x)$' )
x0 = optimize.fminbound(f, - 10 , 0 )
x1 = optimize.fminbound(f, 0 , 5)
print ('x :' , x0, x1)
plt.scatter(x0 , f(x0 ))
plt.scatter(x1 , f(x1 ))
plt.legend(loc=0)
plt.xlabel( '$x$'  )
plt.ylabel( '$y$'  )
plt.grid(True)
plt. show()

solution = optimize.minimize_scalar(f)
print(solution)

print("4.2")
#4.2 Напишите программу, реализующую LU -разложение с выбором
#главного элемента по строке (схема частичного выбора) с выводом матрицы LU и вектора перенумерации переменных (перестановки ст
def LU_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    P = np.eye(n)

    for i in range(n):
        max_row = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[max_row, i]):
                max_row = j
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]

        L[i, i] = 1
        for j in range(i):
            L[i, j] = A[i, j]
            for k in range(j):
                L[i, j] -= L[i, k] * U[k, j]
            L[i, j] /= U[j, j]

        for j in range(i, n):
            U[i, j] = A[i, j]
            for k in range(i):
                U[i, j] -= L[i, k] * U[k, j]

    return L, U, P


def dec_LU(A):
     n=len(A)
     LU=np.copy(A)
     for j in range(0,n-1):
         for i in range(j+1,n):
             if LU[i,j]!=0:
                 u=(LU[i,j])/LU[j,j]
                 LU[i,j+1:n]=LU[i,j+1:n]-u*LU[j,j+1:n]
                 LU[i,j]=u
     return LU

n = 4
Pascal = np.zeros((n, n))
for i in range(n):
    Pascal[i, 0] = 1
    Pascal[0, i] = 1
for i in range(1, n):
    for j in range(1, n):
        Pascal[i, j] = Pascal[i - 1, j] + Pascal[i, j - 1]
print("Matrix Pascal:")
print(Pascal)

L, U, P = LU_decomposition(Pascal)
LU_=np.zeros((n,n))
for i in range(0, n):
    for j in range(0, n):
        for k in range(0, n):
            LU_[i][j] += L[i][k] * U[k][j]
print("Матрица L:")#L — нижняя треугольная матрица с единичной диагональю
print(L)
print("МАтрица U:")#— верхняя треугольная матрица с ненулевыми диагональными элементами

print(U)
print("Матрица P:")#Матрица перестановок
print(P)

print("LU разложение модуль lu:")
print(dec_LU(Pascal))



print("4.4")
#4.4 Рассмотрите алгоритм построения обратной матрицы на основе решения матричного уравнения А Х = Е ,
#где Е —■ единичная, а X — искомая квадратная матрица. Напишите программу вычисления обратной матрицы на основе LU -разлоэюение с выбором
#главного элемента по строке (схема частичного выбора) (задача 4-2). Найдите обратную к матрице Лемера (Lehmer),

def inverse_matrix(A):
    n = len(A)
    L, U, P = LU_decomposition(A)
    E = np.eye(n)
    Y = np.linalg.solve(L, E*P)
    X = np.linalg.solve(U, Y)

    return X
def is_tridiagonal(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(i-j) > 1 and A[i, j] != 0:
                return False
    return True
n=4
Lemmer_matrix = np.zeros((n, n))

for i in range(0,n):
    for j in range(0,n):
            Lemmer_matrix[i, j] = round(min(i+1, j+1) / max(i+1, j+1),1)

A_inv = inverse_matrix(Lemmer_matrix)
print("Lemmer matrix:")
print(Lemmer_matrix)
print("Inverse matrix of Lemmer matrix:")
print(A_inv)
print(np.linalg.inv(Lemmer_matrix))

print("4.5")
#4.5Напишите программу для вычисления числа обусловленности
# квадратной матрицы с использованием норм || • ||„, а = 1,Е ,ос (см. задачу 4Л) и вычисления обратной матрицы на основе LU -разлоэюен
from scipy.linalg import lu, norm


def cond_number(A):
    norm_A = norm(A,ord=1)
    norm_A_inf = norm(A,ord=2)
    norm_A3=norm(A,ord=inf)


    print(f"Норма 1  {norm_A}")
    print(f"Норма 2 {norm_A_inf}")
    print(f"Норма 3 {norm_A3}")


for n in range(2, 6):
    B = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            B[i, j] = min(i + 1, j + 1)
    print("n =", n)
    print("Матрица A:")
    print(B)

    print()
    L, U, P = LU_decomposition(B)
    print()
    print("Матрица L:")  # L — нижняя треугольная матрица с единичной диагональю
    print(L)
    print("МАтрица U:")  # — верхняя треугольная матрица с ненулевыми диагональными элементами
    print(U)
    print()
    print("LU:")
    print(dec_LU(B))
    print("Обратная матрица")
    print(inverse_matrix(B))
    print()
    print(cond_number(B))

def check_anomaly(data):
    mean = np.mean(data) #среднее значение
    std = np.std(data)#стандартное отклонение
    lower_bound = mean - 3 * std #диапазон допустимых значений
    upper_bound = mean + 3 * std
    anomalies = []
    for x in data:
        if x < lower_bound or x > upper_bound:
            anomalies.append(x)
    return anomalies


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
anomalies = check_anomaly(data)
print("Выборка:", data)
if anomalies:
    print("Аномальные значения:", anomalies)
else:
    print("Аномальных значений нет.")
print("9.1")
#9.1 Напишите программу для пахоэюдения значения интерполирующего полинома в точке х на основе рекуррентных соотношений (алгоритм
#Невилля)
#С помощью этой программы найдите значения в точках х = 1.5, 2.5

def neville(x, xs, ys):
    n = len(xs)
    table = np.zeros((n, n))
    for i in range(n):
        table[i, 0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = ((x - xs[i + j]) * table[i, j - 1] - (x - xs[i]) * table[i + 1, j - 1]) / (xs[i] - xs[i + j])
    return table[0, n - 1]


n = 10
a = -3
b = 3
x1 = 1.5
x2 = 2.5
f  = lambda x: (np.arctan(1 + x**2))**(-1)


xs = np.linspace(a, b, n)
ys = f(xs)

y1 = neville(x1, xs, ys)
y2 = neville(x2, xs, ys)


print("Значение интерполирующего полинома в точке", x1, "равно", y1)
print("Значение интерполирующего полинома в точке", x2, "равно", y2)


print("4.3")
#9.3 3 Напишите программу для приблиэ/сения сеточной функции уи
#i = 0 ,1,...,п полиномом рт(х) = с0 щх -Ь 4"  т, n < п методов
#наименьших квадратов.

n = 10
a = 0
b = 2
f = lambda x: 1 - np.cos(x)
h = (b - a) / n

xs = np.linspace(a, b, n)
ys = f(xs)

def approx(xs, ys, m):
    n = len(xs)
    A = np.zeros((n, m + 1))
    for i in range(n):
        for j in range(m + 1):
            A[i, j] = xs[i]**j
    b = ys.reshape((n, 1))
    c = np.linalg.inv(A.T @ A) @ A.T @ b
    return lambda x: sum(c[i, 0] * x**i for i in range(m +1))


p1 = approx(xs, ys, 1)
p2 = approx(xs, ys, 2)
p3 = approx(xs, ys, 3)

x = np.linspace(1, 10, 100)
y = f(x)
plt.plot(x, y, label='y')
plt.plot(x, p1(x), label='p1')
plt.plot(x, p2(x), label='p2')
plt.plot(x, p3(x), label='p3')
plt.legend()
plt.show()


# Напишите программу для приблиэюения сеточной функции yi}
# i = 0 ,1,...,n функцией £(.т) = аеЬх методом наименьших квадратов.
print("9.4")
n = 10
x = np.linspace(0,n, 100)
y = 1 - np.cos(x)
sum_xi_yi = np.sum(x*y)
sum_xi = np.sum(x)
sum_yi = np.sum(y)
sum_xi_squared = np.sum(x**2)
b = (n*sum_xi_yi - sum_xi*sum_yi) / (n*sum_xi_squared - sum_xi**2)
a = (sum_yi - b*sum_xi) / n
ae=np.exp(b)
b=a
f = ae*np.exp(b*x)

plt.plot(x, y, 'o', label='y')
plt.plot(x, f, label='f')
plt.legend()
plt.show()
print(a)
print(b)