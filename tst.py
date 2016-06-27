import numpy as np

from itertools import product, count

for i in count():
    print i



hor_str, ver_str = {-1: "W", 0: "", 1: "E"}, {-1: "N", 0: "", 1: "S"}
dir_str = {(x,y): ver_str[y] + hor_str[x] for x, y in product(hor_str, ver_str)}
print dir_str



# output = np.array([[1],[2],[3]])
# binary_output = 1 * (output == np.max(output))
# print binary_output
# target = np.array([[0],[0],[1]])
# print all(binary_output == target)
#
# a = np.array([1,2,3])
# b = np.copy(a)
# b[0] = 5
# print a, b
# M = np.eye(3)
# N = M
# M += np.array([[1,0,0],[0,0,0],[0,0,0]])
# print M
# print N
# """DETTA FORKLARAR NOG EN HEL DEL!!! (MUTABILITET)"""

# a = np.array([[1],[10],[100]])
# b = a.reshape((1,3))
# print a
# print b
# e = np.array([[1,4],
#      [2,5],
#      [3,6]])
# a = np.array([[10,100],
#      [100,10],
#      [1,2]])
#
# gradient_weights = [None] + [np.dot(e, a.T) for layer in range(1, 2)]
# gradient_biases = [None] + [np.dot(e, np.ones((2, 1))) for layer in range(1, 2)]
#
# print gradient_weights
# print gradient_biases
#
# gradient_weights1 = [None] + [np.outer(e[:,0], a[:,0]) for layer in range(1, 2)]
# gradient_biases1 = [None] + [e[:,0] for layer in range(1, 2)]
#
# gradient_weights2 = [None] + [np.outer(e[:,1], a[:,1]) for layer in range(1, 2)]
# gradient_biases2 = [None] + [e[:,1] for layer in range(1, 2)]
#
# for i in range(1,2):
#      print gradient_biases1[i] + gradient_biases2[i]
#      print gradient_weights1[i] + gradient_weights2[i]
