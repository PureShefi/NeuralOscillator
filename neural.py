import random
import math
import numpy as np
from rle_parser import RunLengthEncodedParser

IS_OSCILLATOR_LAYOR = (1,0)
NOT_OSCILLATOR_LAYOR = (0,1)

class GolDataSet():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def normalize_data(self, data):
        # Make sure that the size is legal
        if rle_parser.size_x > self.size_x or rle_parser.size_y > self.size_y:
            return None

        # Fit all sets into the same size
        p = np.array(data.pattern_2d_array)
        p = np.append(p, [[0 for x in range(self.size_y - rle_parser.size_y)]], 0)
        p = np.append(p, [[0] for x in range(self.size_x - rle_parser.size_x)], 1)

        # Return a single array of the table (Each param will be an input)
        return p.flatten()

    def get_folder_games(self, path):
        """ Get all RLE games in folder by path """
        games = []
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                # Ignore files that we cant parse
                if not file.endswith(".rle"):
                    continue

                data = ""
                file_path = root + file
                with open(file_path, "r") as f:
                    data = f.read()

                # Parses the RLE data and push the 2d array
                rle_parser = RunLengthEncodedParser(data)

                # Normalize all the data to the same size
                data = self.normalize_data(rle_parser)
                if data == None:
                    continue

                games.append(data)

        return games

    def get_training_data_set(self, base_path="training"):
        """
            Return all the games in the testing directory
            in the
        """
        return  [(x, IS_OSCILLATOR_LAYOR)  for x in self.get_folder_games(base_path + "/positive")] +
                [(x, NOT_OSCILLATOR_LAYOR) for x in self.get_folder_games(base_path + "/negative")]


class NeuralNetwork():
    """
        Class containg the neural network logic.
        Should hold all the inputs/outputs and hidden layers
    """
    def __init__(self, network_layers, folds, learning_rate, dataset):
        self.folds = folds
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.initialize_network(network_layers)
        self.train()

    def initialize_network(self, network_layers):
        """ Setup network based on given size """
        self.network = []
        prev = network_layers[0]
        for layer_size in network_layers[1:]:
            layer = [{'weights':[random() for i in range(prev)], 'bias': 0} for i in range(layer_size)]
            self.network.append(layer)
            prev = layer_size

    def activate(self, weights, inputs, bias):
        """ Calculate the value of activation of a neuron based on its inputs """
        return sum([x*y for x,y in zip(weights, inputs)]) + bias

    def transfer(self, activation):
        """ Transfer forward the neuron activation """
        return 1.0 / (1.0 + math.exp(-activation))

    def forward_propagate(self, inputs):
        """ Pass the data through the neural network """
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs, neuron["bias"])
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward_propagate(expected):
        """ Run on all the layer, each one looking at the one infront of it to see its error """
        # Calculate initial error of the output layer
        for neuron, predicted in zip(layer, expected):
            error = neuron['output'] - predicted
            neuron['delta'] = error * transfer_derivative(neuron['output'])

        # Run on the previous layers. Set the delta for each neuron based on the error from the next layer
        for i, layer in reversed(enumerate((self.network[:-1]))):
            for j in range(len(layer)):
                # Gets the sum of the errors of the connected next layer neurons
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i+1]])
                neuron['delta'] = error * transfer_derivative(neuron['output'])

    def update_weights(input_value):
        """ Update the network weights with the error for the back propagate """
        for i, layer in enumerate(self.network):
            inputs = input_value[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in layer:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= self.learning_rate * neuron['delta'] * inputs[j]
                neuron['bias'] -= self.learning_rate * neuron['delta']

    def train(self, epochs):
        """ Train the dataset for epochs count """
        for i in range(epochs):
            for input_value, expected in self.dataset:
                outputs = forward_propagate(input_value)
                backward_propagate(expected)
                update_weights(input_value)

dataset = GolDataSet(25, 25).get_training_data_set()
network = NeuralNetwork()