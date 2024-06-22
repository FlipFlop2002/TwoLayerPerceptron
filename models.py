import numpy as np

class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_units, output_size, seed):
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        np.random.seed(seed)

        # hidden layer
        self.layer1_weights = np.random.randn(input_size, hidden_units)
        self.layer1_bias = np.zeros((1, hidden_units))

        # output layer
        self.layer2_weights = np.random.randn(hidden_units, output_size)
        self.layer2_bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        x = np.dot(x, self.layer1_weights) + self.layer1_bias
        self.hidden_layer_output = x
        x = self.sigmoid(x)
        self.sigm_output = x
        x = np.dot(x, self.layer2_weights) + self.layer2_bias
        self.output = x
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def predict(self, X):
        return np.array([self.forward(x) for x in X]).squeeze(1).squeeze(1)

    def sigmoid_derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2


    def mean_squared_error_derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred)


    def backward(self, y_true, y_pred, lr):
        # hidden layer weights
        layer1_weights_derivative = -2 * (y_true - y_pred) * self.layer2_weights.T * self.sigmoid_derivative(self.hidden_layer_output) * self.input

        # hidden layer bias
        layer1_bias_derivative = -2 * (y_true - y_pred) * self.layer2_weights.T * self.sigmoid_derivative(self.hidden_layer_output) * 1

        # output layer weights
        layer2_weights_derivative = -2 * (y_true - y_pred) * self.sigm_output.T

        # output layer bias
        layer2_bias_derivative = -2 * (y_true - y_pred) * 1



        self.layer2_weights -= lr * layer2_weights_derivative
        self.layer2_bias -= lr * layer2_bias_derivative
        self.layer1_weights -= lr * layer1_weights_derivative
        self.layer1_bias -= lr * layer1_bias_derivative