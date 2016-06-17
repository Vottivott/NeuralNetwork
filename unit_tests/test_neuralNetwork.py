from unittest import TestCase
import neural_network
import neural_network_2
import numpy as np



class TestNeuralNetwork2(TestCase):
    def setUp(self):
        weights, biases, layer_sizes, L, mini_batch = neural_network.load_from_file("test_data.pkl")
        self.neural_network = neural_network.NeuralNetwork(layer_sizes)
        self.neural_network.weights = list(weights)
        self.neural_network.biases = list(biases)
        self.mini_batch = list(mini_batch)

        self.neural_network_2 = neural_network_2.NeuralNetwork(layer_sizes)
        self.neural_network_2.weights = [None] + [np.copy(weight) for weight in weights[1:]]
        self.neural_network_2.biases = [None] + [bias.reshape((len(bias), 1)) for bias in biases[1:]]
        self.mini_batch_2 = [(input.reshape((len(input), 1)),
                             target.reshape((len(target), 1)))
                             for input, target in mini_batch]

    def check_networks_equal(self):
        biases1 = [None] + [bias.reshape((len(bias),1)) for bias in self.neural_network.biases[1:]]
        biases2 = self.neural_network_2.biases

        weights1 = self.neural_network.weights
        weights2 = self.neural_network_2.weights

        diff_b = [biases2 - biases1 for biases2, biases1 in
                  zip(biases2[1:], biases1[1:])]
        diff_w = [weights2 - weights1 for weights2, weights1 in
                  zip(weights2[1:], weights1[1:])]
        print "!"
        for bias1, bias2 in zip(biases1[1:], biases2[1:]):
            for i in range(len(bias1)):
                self.failUnlessAlmostEqual(bias1[i], bias2[i])
        for w1, w2 in zip(weights1[1:], weights2[1:]):
            for weight1, weight2 in zip(np.nditer(w1, order='C'), np.nditer(w2, order='C')):
                self.failUnlessAlmostEqual(weight1, weight2)


    def test_update_mini_batch(self):
        self.check_networks_equal()

        self.neural_network.update_mini_batch(self.mini_batch, 300)
        self.neural_network_2.update_mini_batch(self.mini_batch_2, 300)

        self.check_networks_equal()