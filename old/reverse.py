import numpy as np

import network
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

neural_net = network.Network()
neural_net.load()
target = neural_net.activation_pattern(3)

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

# x = np.linspace(0, 27, 28)
# y = np.linspace(0, 27, 28).reshape(-1, 1)

im = plt.imshow(np.eye(28), cmap=cm.Greys_r, animated=True)


def updatefig(*args):
    neural_net.step_reverse(target,0.05)
    # print neural_net.activities[0]
    im.set_array(1-neural_net.activities[0].reshape((28, 28)))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()



exit(0)

plt.ion()




# image = plt.imshow(1-neural_net.activities[0].reshape((28, 28)), cmap = cm.Greys_r)
for i in range(100):
    neural_net.step_reverse(target,0.05)
    plt.clf()
    plt.imshow(i-neural_net.activities[0].reshape((28, 28)), cmap = cm.Greys_r)
    plt.imshow(i/100.0-np.eye(28), cmap = cm.Greys_r)
    plt.pause(0.05)
    # image.set_data(1-neural_net.activities[0].reshape((28, 28)))
    # image.set_data(np.ones((28,28)))
    # plt.draw()
    # plt.pause(0.05)
    print i
    # time.sleep(0.05)
# classified_digit = neural_net.classify(data)

def get_indefinite_article(classified_digit):
    return "an" if classified_digit == 8 else "a"

# print "Handwritten digit classified as " + get_indefinite_article(classified_digit) + " " + str(classified_digit)

