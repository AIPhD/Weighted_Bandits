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


def multiple_alpha_regret_plots(regrets, alphas):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Number of episodes')
    plt.ylabel(r'$\frac{\mathrm{Regret}}{T}$')
    
    for regret in regrets:
        plt.plot(np.cumsum(regret)/np.cumsum(np.ones(len(regret))), label=r"$\alpha$ = {alpha}".format(alpha=alphas[i]))
        i += 1
    
    plt.legend()
    plt.savefig('/home/steven/weighted_bandits/plots/regret_alpha_comparison.png')

    plt.show()
    plt.close()


def multiple_beta_regret_plots(regrets, bethas=None, plot_label=None, plotsuffix='regret_beta_comparison'):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Episodes')
    plt.ylabel(r'$\frac{\mathrm{Regret}}{T}$')
    
    if bethas is None:
        
        for regret in regrets:
            plt.plot(np.cumsum(regret)/np.cumsum(np.ones(len(regret))), label=plot_label)
            i += 1

    else:

        for regret in regrets:
            plt.plot(np.cumsum(regret)/np.cumsum(np.ones(len(regret))), label=plot_label+r" $\beta$ = {beta}".format(beta=bethas[i]))
            i += 1
    
    plt.legend()
    # plt.savefig('/home/steven/weighted_bandits/plots/'+plotsuffix+'.png')

    # plt.close()
    


def alpha_plots(alphas, betas=None):
    
    plt.xlabel('Episode')
    plt.ylabel(r'$\mathrm{\alpha}$')
    
    i = 0

    if betas is None:

        for alpha in alphas:
            plt.plot(alpha)    

    else:

        for alpha in alphas:
            plt.plot(alpha, label=r"$\beta$ = {beta}".format(beta=betas[i]))
            i += 1
    
    plt.legend()
    plt.savefig('/home/steven/weighted_bandits/plots/alpha_evol.png')

    plt.show()
    plt.close()
    
    
