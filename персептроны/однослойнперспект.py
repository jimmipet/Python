import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.input_size = input_size  # количество нейронов во входном слое
        self.lr = lr  # скорость обучения
        self.epochs = epochs  # количество эпох обучения
        # инициализируем веса случайным образом
        self.W = np.random.randn(self.input_size)
        # инициализируем смещение случайным образом
        self.b = np.random.randn(1)

    def activation_fn(self, x):  # функция активации (ступенчатая)
        return np.where(x > 0, 1, 0)

    def forward(self, x):  # прямое распространение
        # взвешенная сумма входных данных и весов
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation_fn(self.z)  # активация нейрона
        return self.a

    def backward(self, x, y, output):  # обратное распространение ошибки
        error = y - output  # ошибка
        self.W += self.lr * error * x  # обновляем веса
        self.b += self.lr * error  # обновляем смещение
        print(error)

    def train(self, X, y):  # обучение перцептрона
        for epoch in range(self.epochs):  # проходим по каждой эпохе обучения
            # проходим по каждому примеру в обучающей выборке
            for i in range(X.shape[0]):
                x = X[i]  # текущий входной пример
                output = self.forward(x)  # прямое распространение
                # обратное распространение ошибки
                self.backward(x, y[i], output)

    def predict(self, X):  # предсказание на новых данных
        predictions = []
        for i in range(X.shape[0]):
            x = X[i]
            output = self.forward(x)
            predictions.append(output)
        return predictions


p = Perceptron(input_size=2)

# Задаем обучающую выборку
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # входные данные
y = np.array([0, 0, 0, 1])  # соответствующие выходные классы

# Обучаем перцептрон
p.train(X, y)

# Предсказываем на новых данных
new_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = p.predict(new_X)
print(predictions)  # [0, 0, 0, 1]
