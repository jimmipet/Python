import math

class Perception:
    def __init__(self):
        self.w1 = 1  # вес
        self.w2 = 1  # вес
        self.w3 = 1 #
        self.bias = -1.5  # смещение


    def feed_forward(self, x1, x2, x3):
        weighted_sum = (x1 * self.w1) + (x2 * self.w2)+(x3*self.w3) + self.bias
        # использование гиперболического тангенса в качестве активационной функции
        activation = math.tanh(weighted_sum)
        if activation > 0:
            return 1
        else:
            return 0

p = Perception()

print(p.feed_forward(1, 0,1))
print(p.feed_forward(0, 1,1))
print(p.feed_forward(1, 0,0))  
print(p.feed_forward(1, 1,1)) 

