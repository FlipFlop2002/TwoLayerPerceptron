import numpy as np

print(np.random.randn(4, 1).shape)

def sigmoid_derivative(x):
        return np.exp(-x) / (1 + np.exp(-x))**2



X = np.array([[2.3], [3.4], [0.5]])
Y = np.array([[3.3, 5.4, 1.5]])
Y_pred = np.array([[4.3, 8.4, 8.5]])
print(sigmoid_derivative(X))
print(sigmoid_derivative(2.3))
print(sigmoid_derivative(3.4))
print(sigmoid_derivative(4.5))

print("dddddd")
print(X.shape)
print(X.T.shape)
print("TEST")

from models import TwoLayerPerceptron
per = TwoLayerPerceptron(1, 4, 1, 42)
per.forward(3)
per.backward(3, 5, 0.01)

X2 = np.array([[2.3], [3.4], [4.5]])

print(2*X2)

X = np.array([[2, 3, 12]])
e = np.array([[2, 3, 12]])
print(2*e*X)
