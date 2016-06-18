import numpy as np
import pickle
import math
import random
import gzip
from time import time
import winsound

import neural_network
"""
This version will try to do calculations for all training cases in a mini batch at the same time to improve efficiency.
Compare this result with the time for a non-optimized network:
    network = NeuralNetwork((784, 100, 10))
    network.SGD(train_set, 10, 1, 3, test_set, highest_activation)
    ->
    Epoch 0: 9500 out of 10000 test cases correct (95.0%)
    Time elapsed: 54.5859999657 seconds
"""

"""
CHANGES MADE:
    Make self.biases a list of (layer_size, 1) matrices instead of (layer_size) vectors
    Make activity, error and z (layer_size, len(mini_batch)) matrices instead of (layer_size) vectors

"""

"""
What??
Epoch 0: 5002 out of 10000 test cases correct (50.02%)

Time elapsed: 20.4639999866 seconds

Epoch 0: 2007 out of 10000 test cases correct (20.07%)

Time elapsed: 19.0779998302 seconds

1028

980

1028

983

1031

1028

Epoch 0: 2711 out of 10000 test cases correct (27.11%)

1010

mini batch size = 1 ger
Epoch 0: 7968 out of 10000 test cases correct (79.68%)
Time elapsed: 61.3570001125 seconds

jmfr med neural_network (1)
Epoch 0: 7506 out of 10000 test cases correct (75.06%)
Time elapsed: 49.9800000191 seconds

dvs samma
"""

"""
batch_size = 2 ger
Epoch 0: 8877 out of 10000 test cases correct (88.77%)

Time elapsed: 42.9709999561 seconds

batch_size = 5 ger
Epoch 0: 9335 out of 10000 test cases correct (93.35%)

Time elapsed: 25.5700001717 seconds

"""

# TODO: Error is probably caused by training data having the form of (length) vectors instead of (length, 1) matrices??

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [None] + [random_matrix(layer_sizes[i], layer_sizes[i-1], 0.2) for i in range(1, len(layer_sizes))]
        self.biases = [None] + [random_matrix(layer_sizes[i], 1, 0.2) for i in range(1, len(layer_sizes))]
        self.activation_function = np.vectorize(sigmoid)
        self.activation_function_prime = np.vectorize(sigmoid_prime)
        self.cost_function_gradient = least_squares_cost_function_gradient

        self.saved_activities = []

    def SGD(self, training_data, mini_batch_size, epochs, learning_rate, test_data=None, test_evaluation_function=None):
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                correct_outputs = self.test(test_data, test_evaluation_function)
                # save_to_file(self, "saved_networks/" + str(correct_outputs) + ".pkl")
                print "Epoch " + str(i) + ": " + test_result_string(correct_outputs, test_data)
            else:
                print "Epoch " + str(i)
            print

    """ Return the number of correct outputs from the input data"""
    def test(self, test_data, evaluation_function):
        correct_outputs = 0
        for input, target in test_data:
            output = self.feedforward(input)
            if evaluation_function(target, output) == True:
                correct_outputs += 1
        return correct_outputs

    def update_mini_batch(self, mini_batch, learning_rate):
        factor = - float(learning_rate) / len(mini_batch)
        # print "factor (2) = " + str(factor)

        gradient_weights, gradient_biases = self.backpropagation(mini_batch)
        for layer in range(1, self.L):
            self.weights[layer] += factor * gradient_weights[layer]
            self.biases[layer] += factor * gradient_biases[layer]

    def backpropagation(self, mini_batch):
        inputs, targets = map(np.hstack, zip(*mini_batch))

        assert_index = 0
        assert all(inputs[:, assert_index:assert_index+1] == zip(*mini_batch)[0][assert_index])

        activity = [np.zeros((layer_size, len(mini_batch))) for layer_size in self.layer_sizes]
        error = [None] + [np.zeros((layer_size, len(mini_batch))) for layer_size in self.layer_sizes[1:]]
        z = [None] + [np.zeros((layer_size, len(mini_batch))) for layer_size in self.layer_sizes[1:]]

        # 1. Input activation
        activity[0] = inputs


        # 2. Feed-forward
        for layer in range(1, self.L):
            z[layer] = np.dot(self.weights[layer], activity[layer-1]) + self.biases[layer]
            activity[layer] = self.activation_function(z[layer])

            # Z = np.dot(self.weights[layer], activity[layer - 1][:,assert_index]) + self.biases[layer][:,0]
            # # a = self.activation_function(Z)
            # af = np.vectorize(neural_network.sigmoid)
            # a = af(Z)
            # c_a = activity[layer][:,assert_index]
            # assert all(abs(c_a - a) < 0.0000001)


        # 3. Calculate error for last layer
        cost_gradient = self.cost_function_gradient(targets, activity[-1])
        error[-1] = cost_gradient * self.activation_function_prime(z[-1])

        CG = self.cost_function_gradient(targets[:,assert_index], activity[-1][:,assert_index])
        afp = np.vectorize(neural_network.sigmoid_prime)
        E = CG * afp(z[-1][:,assert_index])
        c_E = error[-1][:,assert_index]
        assert all(abs(c_E - E) < 0.0000001)


        # 4. Feed-backward
        for layer in range(self.L-2, 0, -1):
            error[layer] = np.dot(self.weights[layer+1].T, error[layer+1]) * self.activation_function_prime(z[layer])

            E = np.dot(self.weights[layer + 1].T, error[layer + 1][:,assert_index]) * afp(z[layer][:,assert_index])
            c_E = error[layer][:,assert_index]
            assert all(abs(c_E - E) < 0.0000001)


        # 5. Calculate gradients
        # The gradients from all cases in the mini batch are summed up into one weight matrix and bias vector per layer

        gradient_weights = [None] + [np.dot(error[layer], activity[layer-1].T) for layer in range(1, self.L)]


        # v-- this is the same thing as the above
        # gradient_weights = [None] + [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, len(self.layer_sizes))]
        # for layer in range(1,self.L):
        #     for i in range(len(mini_batch)):
        #         e = error[layer][:,i]
        #         a = activity[layer-1][:,i]
        #         gradient_weights[layer] += np.outer(e, a)


        # gradient_weights = [None] + [np.outer(error[layer], activity[layer - 1]) for layer in range(1, self.L)]



        gradient_biases = [None] + [np.dot(error[layer], np.ones((len(mini_batch), 1))) for layer in range(1, self.L)]

        self.saved_activities = [np.copy(gradient_weights[1])]
        # self.saved_activities = [np.copy(error[1][:, i]) for i in range(len(mini_batch))] #[:, i]


        # TODO: Continue debugging... :/
        # TODO: I really don't get what's wrong? A mini_batch of 5 gives 93% while 10 gives 10% ...???

        # gradient_weights = [None] + [np.outer(error[layer], activity[layer-1]) for layer in range(1, self.L)]
        # gradient_biases = [None] + [error[layer] for layer in range(1, self.L)]

        # GW = [None] + [np.outer(error[layer][:,0], activity[layer - 1][:,0]) for layer in range(1, self.L)]
        # GB = [None] + [error[layer][:,0] for layer in range(1, self.L)]
        # c_GW = gradient_weights#[None] + [gradient_weights[layer] for layer in range(1, self.L)]
        # c_GB = gradient_biases#[None] + [gradient_biases[layer] for layer in range(1,self.L)]
        # diff_GW = c_GW[layer] - GW[layer]
        # diff_GB = c_GB[layer] - GB[layer]
        # for layer in range(1, self.L):
        #     assert all(abs(c_GW[layer] - GW[layer]) < 0.0000001)
        #     assert all(abs(c_GB[layer] - GB[layer]) < 0.0000001)


        return gradient_weights, gradient_biases

    def feedforward(self, activity):
        for layer in range(1, self.L):
            activity = self.activation_function(np.dot(self.weights[layer], activity) + self.biases[layer])
        return activity





def random_matrix(rows, cols, max_value):
    return (np.random.rand(rows,cols) - 0.5) * 2 * max_value

def random_vector(size, max_value):
    return (np.random.rand(size) - 0.5) * 2 * max_value

def save_to_file(object, filename):
    with open(filename, mode="wb") as file:
        pickle.dump(object, file)

def load_from_file(filename):
    with open(filename) as file:
        return pickle.load(file)

def sigmoid(z):
    return 1.0 / (1.0 + math.e**(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def least_squares_cost_function_gradient(target, output):
    return (output - target)

def activation_pattern(digit):
    return np.array([[1*(n == digit) for n in range(10)]]).T

def digit_from_activation_pattern(activation_pattern):
    return max(range(10), key=lambda x: activation_pattern[x])

def highest_activation(target, output):
    binary_output = 1 * (output == np.max(output))
    return all(binary_output == target)

def threshold_at_point_five(target, output):
    binary_output = 1 * (output > 0.5)
    return all(binary_output == target)

def test_result_string(test_result, test_data):
    return str(test_result) + " out of " + str(len(test_data)) + " test cases correct (" + str(100*float(test_result)/len(test_data)) + "%)"

def load_mnist_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
        input_activations = lambda input_digits: map(lambda input: input.reshape((len(input),1)), input_digits)
        target_activations = lambda target_digits: map(lambda digit: activation_pattern(digit), target_digits)
        return zip(input_activations(train_set[0]), target_activations(train_set[1])),\
               zip(input_activations(valid_set[0]), target_activations(valid_set[1])),\
               zip(input_activations(test_set[0]), target_activations(test_set[1]))

if __name__ == "__main__":
    t0 = time()
    train_set, valid_set, test_set = load_mnist_data()
    network = NeuralNetwork((784, 100, 10))
    network.SGD(train_set, 10, 1, 3, test_set, highest_activation)
    # save_to_file(network, "network.pkl")
    t = time() - t0
    print "Time elapsed: " + str(t) + " seconds"
    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)