import numpy as np
import math
import gzip
import pickle
import random
from time import time

"""
(hidden_layer_size, learning_rate, mini_batch_size, number_of_epochs)
(100,0.05,10,5000)Correctly classified 6771 out of 10000 hand-written digits (67.71%)
(500,0.01,20,12000)Correctly classified 6907 out of 10000 hand-written digits (69.07%)
(100,0.01,20,12000)Correctly classified 4388 out of 10000 hand-written digits (43.88%)
(100,0.05,10,12000)Correctly classified 7980 out of 10000 hand-written digits (79.8%)
(100,0.05,10,24000)Correctly classified 8442 out of 10000 hand-written digits (84.42%)
(100,0.05,10,30000)Correctly classified 8565 out of 10000 hand-written digits (85.65%) 762 sec (9178 with new counting)
new counting:
(100,0.08,10,30000)Correctly classified 9279 out of 10000 hand-written digits (92.79%)
(100,0.08,10,60000)Correctly classified 9415 out of 10000 hand-written digits (94.15%)
(100,0.08,10,90000)Correctly classified 9502 out of 10000 hand-written digits (95.02%)
(100,0.08,10,120000)Correctly classified 9559 out of 10000 hand-written digits (95.59%)
(100,0.08,10,200000)Correctly classified 9644 out of 10000 hand-written digits (96.44%)
(100,0.08,10,200000)+(100,0.05,10,100000)Correctly classified 9683 out of 10000 hand-written digits (96.83%)
(100,0.08,10,200000)+(100,0.05,10,100000)+(100,0.02,10,500000)Correctly classified 9728 out of 10000 hand-written digits (97.28%)
"""

"""
double-hidden-layered network
(100,100,0.07,10,5000)Correctly classified 7031 out of 10000 hand-written digits (70.31%)
(100,100,0.07,10,15000)Correctly classified 8946 out of 10000 hand-written digits (89.46%)
(100,100,0.07,10,50000)Correctly classified 9348 out of 10000 hand-written digits (93.48%)
(100,100,0.07,10,100000)Correctly classified 9503 out of 10000 hand-written digits (95.03%)
(100,100,0.07,10,150000)Correctly classified 9598 out of 10000 hand-written digits (95.98%)
(100,100,0.07,10,200000)Correctly classified 9650 out of 10000 hand-written digits (96.5%)
(100,100,0.07,10,250000)Correctly classified 9689 out of 10000 hand-written digits (96.89%)
(100,100,0.07,10,300000)Correctly classified 9716 out of 10000 hand-written digits (97.16%)
(100,100,0.07,10,350000)Correctly classified 9734 out of 10000 hand-written digits (97.34%)
(100,100,0.07,10,400000)Correctly classified 9743 out of 10000 hand-written digits (97.43%)
(100,100,0.07,10,450000)
"""

class Network:
    def __init__(self):
        self.L = 4
        self.input_layer_size = 784
        self.first_hidden_layer_size = 100
        self.second_hidden_layer_size = 100
        self.output_layer_size = 10

        self.learning_rate = 0.07
        self.mini_batch_size = 10
        self.number_of_epochs = 50000

        self.initial_max_weight_size = 0.2

        if __name__ == "__main__":
            with gzip.open('mnist.pkl.gz', 'rb') as f:
                self.train_set, self.valid_set, self.test_set = pickle.load(f)
        # train_x, train_y = train_set

        self.weights = [None,
                        self.random_matrix(self.first_hidden_layer_size, self.input_layer_size),
                        self.random_matrix(self.second_hidden_layer_size, self.first_hidden_layer_size),
                        self.random_matrix(self.output_layer_size, self.second_hidden_layer_size)] # list of matrices, one for each layer
        self.biases = [None,
                       self.random_vector(self.first_hidden_layer_size),
                       self.random_vector(self.second_hidden_layer_size),
                       self.random_vector(self.output_layer_size)] # one bias vector per layer

        self.activities = [np.zeros(self.input_layer_size),
                           np.zeros(self.first_hidden_layer_size),
                           np.zeros(self.second_hidden_layer_size),
                           np.zeros(self.output_layer_size),] # on activity vector per layer

        self.activation_function = np.vectorize(lambda x: 1.0 / (1 + math.e**(-x)))
        self.activation_function_derivative = lambda x: self.activation_function(x) * (1 - self.activation_function(x))
        #self.activation_function_derivative = np.vectorize(lambda x: 1.0 / (1 + math.log(-x)))
        #self.cost_function_gradient_for_a_in_last_layer = lambda x

    def step_reverse(self, target, learning_rate=0.05):
        z = [None,
             np.zeros(self.first_hidden_layer_size),
             np.zeros(self.second_hidden_layer_size),
             np.zeros(self.output_layer_size)] # weighed input vector for each layer
        error = [np.zeros(self.input_layer_size),
                 np.zeros(self.first_hidden_layer_size),
                 np.zeros(self.second_hidden_layer_size),
                 np.zeros(self.output_layer_size)] # error vector for each layer

        # 2. Feedforward
        for layer in range(1, self.L):
            z[layer] = np.dot(self.weights[layer], self.activities[layer-1]) + self.biases[layer]
            self.activities[layer] = self.activation_function(z[layer])

        # 3. Calculate output error
        output = self.activities[-1]
        error[-1] = (output - target) * self.activation_function_derivative(z[-1])

        # 4. Backpropagate the error
        for layer in reversed(range(1,self.L-1)):
            error[layer] = np.dot(self.weights[layer+1].T, error[layer+1]) * self.activation_function_derivative(z[layer])

        # 5. Change the first layer's activations
        gradient_first_layer_activities = np.dot(self.weights[1].T, error[1])
        self.activities[0] += -learning_rate * gradient_first_layer_activities


    def train(self):
        for i in range(self.number_of_epochs):
            selected_training_examples = random.sample(zip(*self.train_set), self.mini_batch_size)
            self.update_mini_batch(selected_training_examples)

    def test(self):
        total = len(self.test_set[0])
        correct = sum(self.test_single_case(test_case) for test_case in zip(*self.test_set))
        print "Correctly classified " + str(correct) + " out of " + str(total) + " hand-written digits (" + str((1.0*correct/total)*100) + "%)"

    def test_single_case(self, test_case):
        input, target = test_case

        output = self.classify(input)

        return output == target
        # return all(output_activation_pattern[i] == self.activation_pattern(target)[i] for i in range(10))

    def random_vector(self, size):
        return (np.random.rand(size) - 0.5) * 2 * self.initial_max_weight_size

    def update_mini_batch(self, training_examples):
        factor = -self.learning_rate/len(training_examples)

        updated_weights = [weight_matrix for weight_matrix in self.weights]
        updated_biases = [bias_vector for bias_vector in self.biases]
        for input, target in training_examples:
            gradient_weights, gradient_biases = self.backprop(input, self.activation_pattern(target))
            for layer in range(1, self.L):
                updated_weights[layer] += factor * gradient_weights[layer]
                updated_biases[layer] += factor * gradient_biases[layer]
        self.weights = updated_weights
        self.biases = updated_biases

    def backprop(self, input, target):
        z = [None,
             np.zeros(self.first_hidden_layer_size),
             np.zeros(self.second_hidden_layer_size),
             np.zeros(self.output_layer_size)] # weighed input vector for each layer
        error = [np.zeros(self.input_layer_size),
                 np.zeros(self.first_hidden_layer_size),
                 np.zeros(self.second_hidden_layer_size),
                 np.zeros(self.output_layer_size)] # error vector for each layer

        # 1. Input x
        self.activities[0] = np.array(input)

        # 2. Feedforward
        for layer in range(1, self.L):
            z[layer] = np.dot(self.weights[layer], self.activities[layer-1]) + self.biases[layer]
            self.activities[layer] = self.activation_function(z[layer])

        # 3. Calculate output error
        output = self.activities[-1]
        error[-1] = (output - target) * self.activation_function_derivative(z[-1])

        # 4. Backpropagate the error
        for layer in reversed(range(1,self.L-1)):
            error[layer] = np.dot(self.weights[layer+1].T, error[layer+1]) * self.activation_function_derivative(z[layer])

        # 5. Return the gradients
        gradient_weights = [None, None, None, None]
        gradient_biases = [None, None, None, None]
        for layer in range(1,self.L):
            gradient_weights[layer] = np.outer(error[layer], self.activities[layer-1])
            gradient_biases[layer] = error[layer]
        return gradient_weights, gradient_biases

    def activation_pattern(self, digit):
        return np.array([1*(n == digit) for n in range(10)])

    def random_matrix(self, rows, cols):
        return (np.random.rand(rows,cols) - 0.5) * 2 * self.initial_max_weight_size

    def save(self):
        for i, weight_matrix in enumerate(self.weights):
            np.save("weights_layer"+str(i), weight_matrix)
        for i, bias_vector in enumerate(self.biases):
            np.save("biases_layer"+str(i), bias_vector)

    def load(self):
        for i in range(1,len(self.weights)):
            self.weights[i] = np.load("weights_layer"+str(i)+".npy")
        for i in range(1,len(self.biases)):
            self.biases[i] = np.load("biases_layer"+str(i)+".npy")

    def classify(self, data):
        if np.shape(data) == (28,28):
            data = np.reshape(data, 784)

        z = [None,
             np.zeros(self.first_hidden_layer_size),
             np.zeros(self.second_hidden_layer_size),
             np.zeros(self.output_layer_size)] # weighed input vector for each layer

        # 1. Input x
        self.activities[0] = np.array(data)

        # 2. Feedforward
        for layer in range(1, self.L):
            z[layer] = np.dot(self.weights[layer], self.activities[layer-1]) + self.biases[layer]
            self.activities[layer] = self.activation_function(z[layer])


        a = self.activities[-1]
        output_activation_pattern = 1*(a > 0.5)

        return max(range(10), key=lambda x: a[x])


if __name__ == "__main__":
    t0 = time()
    neural_network = Network()
    neural_network.load()
    # neural_network.train()
    neural_network.test()
    # neural_network.save()
    t = time() - t0
    print "Time elapsed: " + str(t) + " seconds"
    import winsound
    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)