from neural_network_3 import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import cycle

train_set, valid_set, test_set = load_mnist_data()

data_set = train_set
random.shuffle(data_set)

def get_animation_sequence():
    for image, answer in data_set:
        yield 1-image.reshape(28,28)

animation_sequence = get_animation_sequence()
#
# frames = [image for image in animation_sequence]
#
# frames_it = cycle(frames)

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

im = plt.imshow(np.eye(28), cmap=cm.Greys_r, animated=True)
# im = plt.imshow(0.2+0.6*np.eye(28), cmap=plt.get_cmap('afmhot'), animated=True)

def updatefig(*args):
    im.set_array(animation_sequence.next())
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=300, blit=True)
plt.show()
