import numpy as np

from neural_network_3_dropout import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
from itertools import cycle


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from visualize_weights import visualize_weights


def get_sorted_files_by_modified_date(directory):
    # !/usr/bin/env python
    from stat import S_ISREG, ST_CTIME, ST_MODE
    import os, sys, time

    # path to the directory (relative or absolute)
    dirpath = directory

    # get all entries in the directory w/ stats
    entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path)
               for stat, path in entries if S_ISREG(stat[ST_MODE]))
    # NOTE: on Windows `ST_CTIME` is a creation date
    #  but on Unix it could be something else
    # NOTE: use `ST_MTIME` to sort by a modification date

    # for cdate, path in sorted(entries):
    #     print time.ctime(cdate), os.path.basename(path)
    return [path for stat, path in sorted(entries)]


def get_weight_animation_sequence():
    # nets = sorted(os.listdir("./saved_networks"))
    nets = get_sorted_files_by_modified_date("./saved_networks/")
    print nets
    for net in nets:
        yield load_from_file(net)

rows = 20
cols = 20
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

# im = plt.imshow(np.eye(28), cmap=cm.Greys_r, animated=True)
im = plt.imshow(0.2+0.6*np.eye(28), cmap=plt.get_cmap('afmhot'), animated=True)

def updatefig(*args):
    im.set_array(frames_it.next())
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)
plt.show()

