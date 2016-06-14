import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[10,10,10],[10,100,100],[10,10,10]])

c = np.dot(b, np.ones((3,1)))#np.sum(b, axis=0)
print c


