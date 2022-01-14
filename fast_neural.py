import numpy as np
import random
random.seed(1)

# X = (hours studying, hours sleeping), y = score on test
xAll = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
y = np.array(([92], [86], [89]), dtype=float) # output

# scale units
xAll = xAll/np.amax(xAll, axis=0) # scaling input data
y = y/100 # scaling output data (max test score is 100)

# split data
X = np.split(xAll, [3])[0] # training data
xPredicted = np.split(xAll, [3])[1] # testing data

y = np.array(([92], [86], [89]), dtype=float)
y = y/100 # max test score is 100
print("Y", y)

import neural
print("[*] Loading dataset...")
X, y = neural.GolDataSet(25, 25).get_training_data_set_fast()
tests, test_output = neural.GolDataSet(25, 25).get_training_data_set_fast("tests")
xPredicted = tests[25]
print("OUT", test_output[25])

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 25*25
    self.outputSize = 1
    self.hiddenSize = 3

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize) # (3x1) weight matrix from hidden to output layer
    self.W3 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    self.z4 = np.dot(self.z3, self.W3) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z4) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z3_error = self.o_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z3_delta = self.z3_error*self.sigmoidPrime(self.z3) # applying derivative of sigmoid to z2 error

    self.z2_error = self.z3_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.z3_delta) # adjusting second set (hidden --> output) weights
    self.W3 += self.z3.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled): \n" + str(xPredicted))
    print ("Output: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
    NN.train(X, y)

print("Done")
NN.saveWeights()
NN.predict()