import numpy as np

# np.random.choice([0, 1], size=(10,), p=[1./3, 2./3])
a = np.random.choice([0, 1], size=(10,))
print np.diag(a)