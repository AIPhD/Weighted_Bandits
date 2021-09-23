import numpy as np

DIMENSION = 20
DIMENSION_ALIGN = 14
CONTEXT_SIZE = 60
ALPHA = 0.5
BETA = 0.5
DELTA = 0.1
EPSILON = 1
KAPPA = 0.1
LAMB = 10
EPOCHS = 1500
REPEATS = 50

try:
    GAMMA = 0.1 * np.sqrt(DIMENSION *
                          np.log(1 +
                                 1/(LAMB * DIMENSION) +
                                 2 * np.log(1/DELTA))) + np.sqrt(LAMB)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
