import numpy as np

DIMENSION = 20
DIMENSION_ALIGN = 20
CONTEXT_SIZE = 60
NO_SOURCES = 1
ALPHA = 1/(NO_SOURCES + 1)
BETA = 0.5
DELTA = 0.1
EPSILON = 1
SIGMA = 1
KAPPA = 0.1
LAMB = 1
EPOCHS = 1000
REPEATS = 200

try:
    GAMMA = SIGMA * np.sqrt(DIMENSION *
                            np.log(1 +
                                    1/(LAMB * DIMENSION)) +
                            2 * np.log(1/DELTA)) + np.sqrt(LAMB)

except ZeroDivisionError:
    print("Lambda cannot be zero!")
    GAMMA = 0.1
