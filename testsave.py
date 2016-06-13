import numpy as np

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = np.array([9])
# c = [a,b]
# d = "hej"
import pickle

# with open("saved.pkl", mode='wb') as f:
#     pickle.dump((c, d), f)
class Test:

    def __init__(self):
        self.a = 3

    def set_a(self, a):
        self.a = a

    def save(self, filename):
        with open(filename, mode="wb") as f:
            pickle.dump(self, f)

    def load(self, filename):




with open("saved.pkl") as f:
    c, d = pickle.load(f)

print c
print d