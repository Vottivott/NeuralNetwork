from unittest import TestCase
import neural_network
import neural_network_2
import numpy as np

"""
So what I got out from this is that the biases are calculated correctly, while the weights are completely wrong
^FALSKT - det diffen for biaserna ar lika rorig som for weightsen!!!

Hur kan b vara korrekt samtidigt som b_gradient ar oforandrad??
"""

class TestNeuralNetwork2(TestCase):
    def setUp(self):
        weights, biases, layer_sizes, L, mini_batch = neural_network.load_from_file("test_data.pkl")

        # mini_batch = mini_batch[:1]

        self.neural_network = neural_network.NeuralNetwork(layer_sizes)
        self.neural_network.weights = [None] + [np.copy(weight) for weight in weights[1:]]
        self.neural_network.biases = [None] + [np.copy(bias) for bias in biases[1:]]
        self.mini_batch = list(mini_batch)

        self.neural_network_2 = neural_network_2.NeuralNetwork(layer_sizes)
        self.neural_network_2.weights = [None] + [np.copy(weight) for weight in weights[1:]]
        self.neural_network_2.biases = [None] + [bias.reshape((len(bias), 1)) for bias in biases[1:]]
        self.mini_batch_2 = [(input.reshape((len(input), 1)),
                             target.reshape((len(target), 1)))
                             for input, target in mini_batch]

        acs2 = self.neural_network_2.weights[1:]
        acs1 = self.neural_network.weights[1:]
        diff = [a2 - a1 for a2, a1 in zip(acs2, acs1)]
        print "!"

    def check_networks_equal(self):
        biases1 = [None] + [bias.reshape((len(bias),1)) for bias in self.neural_network.biases[1:]]
        biases2 = self.neural_network_2.biases

        weights1 = self.neural_network.weights
        weights2 = self.neural_network_2.weights

        diff_b = [b2 - b1 for b2, b1 in
                  zip(biases2[1:], biases1[1:])]
        diff_w = [w2 - w1 for w2, w1 in
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


        acs1 = self.neural_network.saved_activities
        acs2 = self.neural_network_2.saved_activities
        # diff = [a2 - a1 for a2, a1 in zip(acs2, acs1)]
        diff_sum = acs2[0] - sum(acs1)#acs2[0].reshape(len(acs2[0])) - sum(acs1)
        diff = [a2 - a1 for a2, a1 in zip(acs2, acs1)]
        print acs1
        print acs2
        self.check_networks_equal()