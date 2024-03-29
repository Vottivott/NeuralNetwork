"""

Vad kunna gora?

- kunna spara och aterskapa natverk sparade i filer, dar all information om natverket finns

- i konstruktorn:
 - ta in en lista over storlekarna pa lagren, tex (784, 100, 100, 10)
 - ta in de olika hypervariablerna

 - ha en metod for att trana, tar in ett dataset som argument

 - ha en metod for att klassificera enligt de aktuella vikterna

- ateranvanda kod mer


"""
import numpy as np
import pickle
import math
import random
import gzip
from time import time
import winsound

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [None] + [random_matrix(layer_sizes[i], layer_sizes[i-1], 0.2) for i in range(1, len(layer_sizes))]
        self.biases = [None] + [random_vector(layer_sizes[i], 0.2) for i in range(1, len(layer_sizes))]
        self.activation_function = np.vectorize(sigmoid)
        self.activation_function_prime = np.vectorize(sigmoid_prime)
        self.cost_function_gradient = least_squares_cost_function_gradient

        self.saved_activities = []

    def SGD(self, training_data, mini_batch_size, epochs, learning_rate, test_data=None, test_evaluation_function=None):
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for j, mini_batch in enumerate(mini_batches):
                # if j == len(mini_batches)/2:
                #     save_to_file((self.weights, self.biases, self.layer_sizes, self.L, mini_batch), "test_data.pkl")
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
        # new_weights = [None] + [np.copy(w) for w in self.weights[1:]]
        # new_biases = [None] + [np.copy(b) for b in self.biases[1:]]
        delta_weights = [None] + [np.zeros_like(w) for w in self.weights[1:]]
        delta_biases = [None] + [np.zeros_like(b) for b in self.biases[1:]]
        # new_weights = [None] + [w for w in self.weights[1:]]
        # new_biases = [None] + [b for b in self.biases[1:]]
        factor = - float(learning_rate) / len(mini_batch)
        # print "factor (1) = " + str(factor)
        for input, target in mini_batch:
            gradient_weights, gradient_biases = self.backpropagation(input, target)
            delta_weights = [None] + [dw + ddw for dw, ddw in zip(delta_weights, gradient_weights)[1:]]
            delta_biases = [None] + [db + ddb for db, ddb in zip(delta_biases, gradient_biases)[1:]]
            # for layer in range(1, self.L):
            #
                # delta_weights[layer] += factor * gradient_weights[layer]
                # delta_biases[layer] += factor * gradient_biases[layer]
                # new_weights[layer] = new_weights[layer] + factor * gradient_weights[layer]
                # new_biases[layer] = new_biases[layer] + factor * gradient_biases[layer]

        # diff_sum = delta_weights[1] - sum(self.saved_activities)

        self.weights = [None] + [w + factor*nw for w,nw in zip(self.weights, delta_weights)[1:]]
        self.biases = [None] + [b + factor*nb for b,nb in zip(self.biases, delta_biases)[1:]]


    def backpropagation(self, input, target):
        activity = [np.zeros(layer_size) for layer_size in self.layer_sizes]
        error = [None] + [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
        z = [None] + [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]

        # 1. Input activation
        activity[0] = input


        # 2. Feed-forward
        for layer in range(1, self.L):
            z[layer] = np.dot(self.weights[layer], activity[layer-1]) + self.biases[layer]
            activity[layer] = self.activation_function(z[layer])


        # 3. Calculate error for last layer
        cost_gradient = self.cost_function_gradient(target, activity[-1])
        error[-1] = cost_gradient * self.activation_function_prime(z[-1])


        # 4. Feed-backward
        for layer in range(self.L-2, 0, -1):
            error[layer] = np.dot(self.weights[layer+1].T, error[layer+1]) * self.activation_function_prime(z[layer])


        # 5. Calculate gradients
        gradient_weights = [None] + [np.outer(error[layer], activity[layer-1]) for layer in range(1, self.L)]
        gradient_biases = [None] + [error[layer] for layer in range(1, self.L)]

        # self.saved_activities.append(np.copy(gradient_weights[1]))  # [-1])

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
    return np.array([1*(n == digit) for n in range(10)])

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
        target_activations = lambda target_digits: map(lambda digit: activation_pattern(digit), target_digits)
        return zip(train_set[0], target_activations(train_set[1])),\
               zip(valid_set[0], target_activations(valid_set[1])),\
               zip(test_set[0], target_activations(test_set[1]))

if __name__ == "__main__":
    t0 = time()
    train_set, valid_set, test_set = load_mnist_data()
    network = NeuralNetwork((784, 100, 10))
    network.SGD(train_set, 10, 1, 3, test_set, highest_activation)
    # save_to_file(network, "network.pkl")
    t = time() - t0
    print "Time elapsed: " + str(t) + " seconds"
    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)