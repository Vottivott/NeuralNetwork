from neural_network import *
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt


def visualize_weights(weights, rows=10, cols=10, neuron_shape=(28,28)):

    # visualization_function = get_linear_visualization_function(network)
    visualization_function = sigmoid

    graph = np.zeros((neuron_shape[0]*rows,neuron_shape[1]*cols))
    for y in range(rows):
        for x in range(cols):
            graph[y*neuron_shape[0]:(y+1)*neuron_shape[0], x*neuron_shape[1]:(x+1)*neuron_shape[1]] = get_weight_visualization(weights[y*cols+x], visualization_function, neuron_shape)

    return graph


def get_linear_visualization_function(weights):
    max_val = weights.max()
    min_val = weights.min()
    return lambda x: (x - min_val) / (max_val - min_val)


def get_weight_visualization(weight_matrix, visualization_function, neuron_shape):
    f = np.vectorize(visualization_function)
    return f(weight_matrix).reshape(neuron_shape)

if __name__ == "__main__":
    net = load_from_file("saved_networks/9720.pkl")
    # data = visualize_weights(net.weights[2], 1, 1, (10, 10))
    data = visualize_weights(net.weights[1], 10, 10, (28, 28))
    plt.imshow(data, cmap = cm.Greys_r)
    plt.show()