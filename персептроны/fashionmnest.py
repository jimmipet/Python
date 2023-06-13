from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки',
           'сумка', 'ботинки']  # Создаём список категорий, на которые будем классифицировать


# Создаём новую фигуру размером 10x10 дюймов и отрисовываем первые 10 элементов набора данных:
plt.figure(figsize=(10, 10))
for i in range(100, 150):
    plt.subplot(5, 10, i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])

x_train = x_train.reshape(60000, 784)  # Меняем размер массива x_train:
# Для получения данных от 0 до 1 нормализуем их путем применения векторизованных операций к каждому элементу массива отдельно:
x_train = x_train / 255

# Преобразуем y_train в бинарную матрицу классов размером 10x10
y_train = utils.to_categorical(y_train, 10)

model = Sequential()  # Создаем последовательную модель
# Добавляем входной полносвязный слой, 128 нейронов, 784 входа в каждый нейрон
model.add(Dense(64, input_dim=784, activation="relu"))
# добавить отсевающий слой, чтобы уменьшить переобучение
model.add(Dropout(0.2))
# Добавляем выходной полносвязный слой, 10 нейронов
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
print(model.summary())

model.fit(x_train, y_train, batch_size=100,
          epochs=10, verbose=1)  # Обучаем модель

predictions = model.predict(x_train)  # Получаем прогноз на обученной модели
n = 10
# Определим класс изображения с номером 1
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

classes[np.argmax(predictions[1])]

