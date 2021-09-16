import numpy as np

dimension = 20
context_size = 60
alpha = 0.5
beta = 0.5

delta = 0.4
epsilon = 0.2
kappa = 0.1
lamb = 1
epochs = 1500
repeats = 50
gamma = np.sqrt(dimension * np.log(1 + 1/(lamb * dimension) + 2 * np.log(1/delta))) + np.sqrt(lamb)
