import random
import math
import numpy as np
import os
import logging
import pickle
from rle_parser import RunLengthEncodedParser

IS_OSCILLATOR_LAYOR = (1,)
NOT_OSCILLATOR_LAYOR = (0,)

MAX_HEIGHT = 25
MAX_WIDTH = 25

HIDDEN_LAYERS = [5]
LEARNING_RATE = 0.3
EPOCHS = 500

# Set random seed so our starting network is always the same
random.seed(1)

# Init our logger
logging.basicConfig(format='[%(levelname)s] %(message)s')

class GolDataSet():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def normalize_data(self, data):
        # Make sure that the size is legal
        if data.size_x > self.size_x or data.size_y > self.size_y:
            return None

        # Center the data
        diff_x = (self.size_x - data.size_x) / 2
        diff_y = (self.size_y - data.size_y) / 2

        padding_x = (math.floor(diff_x), math.ceil(diff_x))
        padding_y = (math.floor(diff_y), math.ceil(diff_y))

        # Fit all sets into the same size
        p = np.array(data.get_board_pattern())
        p = np.pad(p, (padding_y, padding_x), mode='constant', constant_values=0)

        # Return a single array of the table (Each param will be an input)
        return p.flatten()

    def get_folder_games(self, path):
        """ Get all RLE games in folder by path """
        print("[*] Loading from {}".format(path))
        games = []
        not_rle = 0
        failed_loading = 0
        failed_normalizing = 0
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                # Ignore files that we cant parse
                if not file.endswith(".rle"):
                    not_rle += 1
                    continue

                data = ""
                file_path = root + file
                with open(file_path, "r") as f:
                    data = f.read()

                # Parses the RLE data and push the 2d array
                try:
                    rle_parser = RunLengthEncodedParser(data)
                except:
                    failed_loading += 1
                    logging.info("Failed Loading {}".format(file))
                    continue

                # Normalize all the data to the same size
                data = self.normalize_data(rle_parser)
                if data is None:
                    failed_normalizing += 1
                    logging.info("Failed normalizing {}".format(file))
                    continue

                logging.debug("Loaded {}".format(file))
                games.append(data)

            logging.warning("Failed getting presets - loading: {}, normalizing: {}".format(failed_loading, failed_normalizing))
            print("[*] Loaded {}/{} files".format(len(games), len(files)))

        return np.array(games)

    def get_training_data_set(self, base_path="training"):
        """
            Return all the games in the testing directory
            in the
        """
        return  [(x, [1]) for x in self.get_folder_games(base_path + "/positive/")] + \
                [(x, [0]) for x in self.get_folder_games(base_path + "/negative/")]


    def get_training_data_set_fast(self, base_path="training"):
        """
            Return all the games in the testing directory
            in the
        """
        positive = self.get_folder_games(base_path + "/positive/")
        negative = self.get_folder_games(base_path + "/negative/")[:50]
        expected = np.array([np.array([1,0]) for x in positive] + [np.array([0,1]) for x in negative])
        return np.concatenate((positive, negative), axis=0), expected
class NeuralNetwork():
    """
        Class containg the neural network logic.
        Should hold all the inputs/outputs and hidden layers
    """
    def __init__(self, dataset, network_layers, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.network = []
        self.initialize_network(network_layers)
        self.train(epochs)

    def initialize_network(self, network_layers):
        """ Setup network based on given size """
        self.network = []
        prev = network_layers[0]
        for layer_size in network_layers[1:]:
            layer = [{'weights':[random.random() for i in range(prev)], 'bias': 0} for i in range(layer_size)]
            self.network.append(layer)
            prev = layer_size

    def activate(self, weights, inputs, bias):
        """ Calculate the value of activation of a neuron based on its inputs """
        return sum([x*y for x,y in zip(weights, inputs)]) + bias

    def transfer(self, activation):
        """ Transfer forward the neuron activation """
        return 1.0 / (1.0 + math.exp(-activation))

    def transfer_derivative(self, output):
        return output * (1.0 - output)


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

    def backward_propagate(self, expected):
        """ Run on all the layer, each one looking at the one infront of it to see its error """
        # Calculate initial error of the output layer
        output_layer = self.network[-1]
        for neuron, predicted in zip(output_layer, expected):
            error = neuron['output'] - predicted
            neuron['delta'] = error * self.transfer_derivative(neuron['output'])

        # Run on the previous layers. Set the delta for each neuron based on the error from the next layer
        for i, layer in reversed(list(enumerate((self.network[:-1])))):
            for j in range(len(layer)):
                # Gets the sum of the errors of the connected next layer neurons
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in self.network[i+1]])
                layer[j]['delta'] = error * self.transfer_derivative(neuron['output'])

    def update_weights(self, input_value):
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
        print("[*] Starting training...")
        for i in range(epochs):
            for j, (input_value, expected) in enumerate(self.dataset):
                outputs = self.forward_propagate(input_value)
                self.backward_propagate(expected)
                self.update_weights(input_value)
                print("\r[*] Training Epoch {}/{}, Input {}/{}          ".format(i, epochs, j, len(self.dataset)), end="")
        print()

def main():
    if True:
        print("[*] Loading dataset...")
        dataset = GolDataSet(MAX_HEIGHT, MAX_WIDTH).get_training_data_set()

        input_size = MAX_HEIGHT * MAX_WIDTH
        output_size = 1
        layers = [input_size] + HIDDEN_LAYERS + [output_size]
        print("[*] Teaching the network...")
        network = NeuralNetwork(dataset, layers, LEARNING_RATE, EPOCHS)

        with open("db_neural.pickle", "wb") as handle:
            pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("[*] Using precalculated network")
        network = None
        with open("db_neural.pickle", 'rb') as handle:
            network = pickle.load(handle)


    tests = GolDataSet(MAX_HEIGHT, MAX_WIDTH).get_training_data_set("tests")

    input_val, expected = tests[20]
    print(input_val, expected)
    output = network.forward_propagate(input_val)
    print("Output: {}, expected: {}".format(output, expected))

if __name__ == "__main__":
    main()