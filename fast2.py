import numpy as np
import time
import random

random.seed(1)
np.random.seed(10)

class DeepNeuralNetwork():
    def __init__(self, layers, epochs=50, l_rate=0.02):
        self.layers = layers
        self.size = len(layers) - 1
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        params = {"W": np.array([(np.random.randn(self.layers[i+1], self.layers[i]) * np.sqrt(1. / self.layers[i+1])) for i in range(self.size)]),
                  "Z": [0 for x in range(self.size)],
                  "A": [0 for x in range(self.size+1)]}


        print(len(params["W"]))
        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params["A"][-1] = x_train
        prev_neuron = x_train

        for i in range(self.size):
            params['Z'][i] = np.dot(params["W"][i], prev_neuron)
            params['A'][i] = self.sigmoid(params['Z'][i])
            prev_neuron = params['A'][i]

        return prev_neuron

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {"W": [0 for x in range(self.size)]}
        inital_error = 2 * (output - y_train) / output.shape[0]

        for i in reversed(range(self.size)):
            error = inital_error * self.sigmoid(params['Z'][i], derivative=True)
            change_w['W'][i] = np.outer(error, params["A"][i-1])
            inital_error = np.dot(params['W'][i].T, error)

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            for i in range(self.size):
                self.params[key][i] -= self.l_rate * value[i]

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))

import neural
print("[*] Loading dataset...")
x_train, y_train = neural.GolDataSet(25, 25).get_training_data_set_fast()
x_val, y_val = neural.GolDataSet(25, 25).get_training_data_set_fast("tests")

dnn = DeepNeuralNetwork(layers=[25*25, 128, 64, 16, 2], epochs=70)
dnn.train(x_train, y_train, x_val, y_val)

def round2(val):
    return [round(val[0]), round(val[1])]

for x, y in zip(x_val, y_val):
    print("OUTPUT", round2(dnn.forward_pass(x)), y, round2(dnn.forward_pass(x)) == y)