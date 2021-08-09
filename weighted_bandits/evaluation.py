import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import config as c


def regret_plot(regret_evol):
    '''Plot function to showcase regret over all episodes.'''
    plt.plot(regret_evol)
    plt.show()