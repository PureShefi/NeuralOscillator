import numpy as np
import time
import random
import logging
import os
import math
from rle_parser import RunLengthEncodedParser


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

random.seed(1)
np.random.seed(10)

MAX_HEIGHT = 25
MAX_WIDTH = 25

HIDDEN_LAYERS = [128, 64, 16]
LEARNING_RATE = 0.03
EPOCHS = 50

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
            The expected values are returned as another array
        """
        positive = self.get_folder_games(base_path + "/positive/")
        negative = self.get_folder_games(base_path + "/negative/")[:50]
        expected = np.array([np.array([1,0]) for x in positive] + [np.array([0,1]) for x in negative])
        return np.concatenate((positive, negative), axis=0), expected


class NeuralNetwork():
    def __init__(self, layers, epochs=50, learning_rate=0.03):
        self.layers = layers
        self.size = len(layers) - 1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.params = self.initialize_network()

    def sigmoid(self, x, derivative=False):
        """ Calculate the sigmoid value of x. Basically converts all the values to be between 0 and 1 """
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def initialize_network(self):
        """
            Get the initial value of the network parameters
            'Weights': The weight of each connection between the neurons in the layers
            'MidInputs': The values of the weights and neurons before
            'Activation': The output that the nuerons fire
        """
        # Set empty values for all the parameters
        # Set initial weights as a random value which we will fix with backward propagate
        params = {'Weights': np.array([(np.random.randn(self.layers[i+1], self.layers[i]) * np.sqrt(1. / self.layers[i+1])) for i in range(self.size)]),
                  'MidInputs': [0 for x in range(self.size)],
                  'Activation': [0 for x in range(self.size+1)]}

        return params

    def forward_propagate(self, input_value):
        """
            Run the input through our neural network and return the inputs of the last layer
        """

        # Save the initial input for the backward_propagate
        self.params['Activation'][-1] = input_value

        # input layer activations becomes sample
        prev_neurons = input_value

        for i in range(self.size):
            # Matrix multiply the inputs from the neurons that came before and the weight of each one
            # this returns the overall calculated inputs for all the neurons in our layer
            self.params['MidInputs'][i] = np.dot(self.params['Weights'][i], prev_neurons)

            # Pass the inputs through the activation function
            self.params['Activation'][i] = self.sigmoid(self.params['MidInputs'][i])

            # Keep our new values so the next layer can work with it
            prev_neurons = self.params['Activation'][i]

        return prev_neurons

    def backward_propagate(self, expected, output):
        '''
            Backward Propagate our network.
            Compare our output to the expected value.
            Run backwards on the difference and save the needed changes to our network.
            Return the changes
        '''
        # Dictionary to save all the changes we need to set on our network
        change_w = {'Weights': [0 for x in range(self.size)]}

        # Get the difference from the value we got and the value we expected
        inital_error = 2 * (output - expected) / output.shape[0]

        # We need to run from the end to the begging and fix each layer by the values the next layer expects
        for i in reversed(range(self.size)):
            # Get the size of our error
            error = inital_error * self.sigmoid(self.params['MidInputs'][i], derivative=True)

            # Get the matrix of needed changes
            change_w['Weights'][i] = np.outer(error, self.params['Activation'][i-1])

            # Save the layer errors so we can fix the layer before it
            inital_error = np.dot(self.params['Weights'][i].T, error)

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from Stochastic Gradient Descent and
            the given values from the backward propagate
        '''

        # Run on all our needed changes based on the backward propagate
        for key, value in changes_to_w.items():
            for i in range(self.size):
                self.params[key][i] -= self.learning_rate * value[i]

    def compute_accuracy(self, input_values, expected):
        '''
            Check how close our network predicts.
            Runs on the given inputs and returns the mean of how many we succeeded in getting
        '''
        predictions = []

        # Run on all our inputs and compare the values
        for x, y in zip(input_values, expected):
            calculated = self.forward_propagate(x)
            predictions.append(np.argmax(calculated) == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        """
            Train the network based on x_train and expected values y_train
            Validate learning with test values (x_val/y_val)
        """
        print("[*] Starting training")
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_propagate(x)
                network_changes = self.backward_propagate(y, output)
                self.update_network_parameters(network_changes)

            accuracy = self.compute_accuracy(x_val, y_val) * 100
            runtime = time.time() - start_time
            print('\rEpoch: {0:>3}, Time: {1:.2f}s, Accuracy: {2:.2f}%'.format(iteration+1, runtime, accuracy))


def round_output(val):
    return [round(val[0]), round(val[1])]

def main():
    print("[*] Loading dataset...")
    x_train, y_train = GolDataSet(MAX_HEIGHT, MAX_WIDTH).get_training_data_set()
    x_val, y_val = GolDataSet(MAX_HEIGHT, MAX_WIDTH).get_training_data_set("tests")

    print("[*] Creating Neural Network")
    input_size = MAX_HEIGHT * MAX_WIDTH
    layers = [input_size] + HIDDEN_LAYERS + [2]
    dnn = NeuralNetwork(layers=layers, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    dnn.train(x_train, y_train, x_val, y_val)

    print("[*] Training Test:")
    for i, (x, y) in enumerate(zip(x_val, y_val)):
        calcaulated = round_output(dnn.forward_propagate(x))
        print("Set {:>3} [{}]: Expected/Received {}/{}".format(i,
                                                               calcaulated[0] == y[0],
                                                               y,
                                                               calcaulated))

if __name__ == "__main__":
    main()