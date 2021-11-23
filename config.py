import numpy as np

DIMENSION = 20
DIMENSION_ALIGN = 20
CONTEXT_SIZE = 60
NO_SOURCES = 1
ALPHA = 1/(NO_SOURCES + 1)
BETA = 0.1
DELTA = 0.1
EPSILON = 1
SIGMA = 1/np.sqrt(2 * np.pi)
KAPPA = 0.1
LAMB = 1
EPOCHS = 1000
REPEATS = 21

try:
    GAMMA = SIGMA * np.sqrt(DIMENSION *
                            np.log(1 +
                                    1/(LAMB * DIMENSION)) +
                            2 * np.log(1/DELTA)) + np.sqrt(LAMB)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
