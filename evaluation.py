import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import config as c


font  = {
    'size' : 14
}


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

matplotlib.rc('font', **font)

def regret_plot(evol, ylabel='Cummulative regret'):
    '''Plot function to showcase regret over all episodes.'''

    plt.xlabel('Number of episodes')
    plt.ylabel(ylabel)
    plt.plot(evol)
    plt.show()


def multiple_regret_plots(regrets, alphas):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Number of episodes')
    plt.ylabel(r'$\frac{\mathrm{Regret}}{T}$')
    
    for regret in regrets:
        plt.plot(np.cumsum(regret)/np.cumsum(np.ones(len(regret))), label=r"$\alpha$ = {alpha}".format(alpha=alphas[i]))
        i += 1
    
    plt.legend()
    plt.savefig('/home/steven/weighted_bandits/plots/expected_regret.png')

    plt.show()
    plt.close()
    


def alpha_plots(alphas):
    
    plt.xlabel('Number of episodes')
    plt.ylabel(r'$\mathrm{\alpha}$')
    
    for alpha in alphas:
        plt.semilogy(alpha, label=r"$\alpha$ = {alpha}".format(alpha=alpha[0]))
    
    plt.legend()
    plt.savefig('/home/steven/weighted_bandits/plots/alpha_evol.png')

    plt.show()
    plt.close()
    
