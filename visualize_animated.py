import numpy as np

from neural_network_2 import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
from itertools import cycle


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from visualize_weights import visualize_weights

def get_weight_animation_sequence():
    nets = sorted(os.listdir("./saved_networks"))
    for net in nets:
        yield load_from_file("saved_networks/" + net)

rows = 10
cols = 10
shape = (28,28)
layer = 1

# rows = 1
# cols = 10
# shape = (10,10)
# layer = 2

weight_animation_sequence = get_weight_animation_sequence()

frames = [visualize_weights(net.weights[layer], rows, cols, shape) for net in weight_animation_sequence]

frames_it = cycle(frames)

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

im = plt.imshow(np.eye(28), cmap=cm.Greys_r, animated=True)


def updatefig(*args):
    im.set_array(frames_it.next())
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

