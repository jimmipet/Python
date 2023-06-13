import numpy as np

# Определяем размерность матрицы
n = 8

# Создаем матрицу A на основе формулы 
a = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a[i, j] = (i+j+1) * np.math.factorial(i+j) / (np.math.factorial(i) * np.math.factorial(j) * np.math.factorial(i+j+1))

# Вычисляем три наименьших собственных значения и соответствующие им собственные векторы
# исопльзуем встроенную функцию eig для вычисления собственных векторов матрицы 
eigenvalues, eigenvectors = np.linalg.eig(a)
# выбираем только первые 3 наименьших значения собственных векторов 
idx = eigenvalues.argsort()[:3]
# записываем в массивы данных
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Выводим результаты
print("Три наименьших собственных значения:")
print(eigenvalues)
print("Соответствующие собственные векторы:")
print(eigenvectors)
