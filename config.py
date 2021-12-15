import numpy as np

DIMENSION = 35
DIMENSION_ALIGN = 35
CONTEXT_SIZE = 25
NO_SOURCES = 10
ALPHA = 1/(NO_SOURCES + 1)
BETA = 0.1
DELTA = 0.1
EPSILON = 1
SIGMA = 1/np.sqrt(2 * np.pi)
KAPPA = 0.1
LAMB = 1
EPOCHS = 200
REPEATS = 99

try:
    GAMMA = SIGMA * np.sqrt(DIMENSION *
                            np.log(1 +
                                    1/(LAMB * DIMENSION)) +
                            2 * np.log(1/DELTA)) + np.sqrt(LAMB)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
