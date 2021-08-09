import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import config as c


def regret_plot(evol, ylabel):
    '''Plot function to showcase regret over all episodes.'''

    plt.xlabel('Number of episodes')
    plt.ylabel('Cummulative regret')
    plt.plot(evol)
    plt.show()


def multiple_regret_plots(regrets, alphas):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Number of episodes')
    plt.ylabel('Cummulative regret')
    
    for regret in regrets:
        plt.plot(np.cumsum(regret), label="alpha = {alpha}".format(alpha=alphas[i]))
        i += 1
    
    plt.legend()
    plt.show()
    
    
