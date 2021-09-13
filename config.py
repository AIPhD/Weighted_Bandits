import numpy as np

dimension = 10
context_size = 100
alpha = 0.5
beta = 0.1

delta = 0.4
epsilon = 0.1
kappa = 0.2
lamb = 1
epochs = 2000
repeats = 200
gamma = np.sqrt(dimension * np.log(1 + 1/(lamb * dimension) + 2 * np.log(1/delta))) + np.sqrt(lamb)
