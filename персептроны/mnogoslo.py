import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


a, b = 3, 2
x0, y0 = 0, 0
num_points = 10
X = np.random.uniform(-6, 6, (num_points, 2))
y = np.zeros(num_points)


for i in range(num_points):
    distance_to_center = np.sqrt((X[i, 0] - x0) ** 2 / a ** 2 +
                                 (X[i, 1] - y0) ** 2 / b ** 2)
    y[i] = 1 if distance_to_center <= 1 else 0


model = MLPClassifier(hidden_layer_sizes=(64, 64), 
                      activation='relu',
                      solver='adam', max_iter=100) 


from sklearn.linear_model import Perceptron
model = Perceptron(max_iter=100)
model.fit(X, y)


accuracy = model.score(X, y)
print('Accuracy: %.2f' % (accuracy * 100))

for i in range(num_points):
    label = 'inside' if y[i] == 1 else 'outside'
    print(f'Point {i}: ({X[i][0]}, {X[i][1]}), Label: {label}')