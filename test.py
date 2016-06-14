import numpy as np

# o = np.ones((3,1))
# print o
# a = np.array([[1,2,3]]).T
# b = np.array([[4,5,6]]).T
# print np.hstack((a,b))

# a = np.array([[2,1,0],[3,2,0],[0,0,2]])
# b = np.array([[1,2,3,10],[4,5,6,10],[7,8,9,10]])
#
# c = np.array([[10,100,1000]]).T
# print a
# print a[:, 1:2]

a = np.array([1,2,3])
b = np.array([2,2,5])

print a.shape
print a.dot(b)