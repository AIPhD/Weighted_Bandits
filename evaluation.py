import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import config as c
matplotlib.use('TKAgg')

font  = {
    'size' : 14
}

PLOT_DIR = '/home/steven/weighted_bandits/plots/'

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
        alpha=alphas[i]
        plt.plot(regret/np.cumsum(np.ones(len(regret))),
                 label=f"$\alpha$ = {alpha}")
        i += 1

    plt.legend()
    plt.savefig('{PLOT_DIR}/regret_alpha_comparison.png')
    plt.show()
    plt.close()


def multiple_beta_regret_over_time_plots(regrets,
                                         errors,
                                         bethas=None,
                                         plot_label=None,
                                         plotsuffix='',
                                         directory='',
                                         do_plot=False):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Episodes')
    plt.ylabel(r'$\frac{\mathrm{Regret}}{T}$')
    plt.ylim(0, 2)

    if bethas is None:
        for regret in regrets:
            yerr = errors[i]/np.cumsum(np.ones(len(regret)))
            plt.fill_between(np.cumsum(np.ones(c.EPOCHS)),
                             regret/np.cumsum(np.ones(len(regret))) + yerr,
                             regret/np.cumsum(np.ones(len(regret))) - yerr,
                             label=plot_label,
                             alpha=0.25)
            i += 1

    else:
        for regret in regrets:
            yerr = errors[i]/np.cumsum(np.ones(len(regret)))
            beta=bethas[i]
            plt.fill_between(np.cumsum(np.ones(c.EPOCHS)),
                             regret/np.cumsum(np.ones(len(regret))) + yerr,
                             regret/np.cumsum(np.ones(len(regret))) - yerr,
                             label=f"{plot_label}" + r" $\beta$" + f" = {beta}",
                             alpha=0.25)
            i += 1

    plt.legend()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/'+
                    plotsuffix+
                    'regret_over_time_beta_comparison.png')
        # plt.show()
        plt.close()



def multiple_beta_regret_plots(regrets,
                               errors=None,
                               bethas=None,
                               plot_label=None,
                               plotsuffix='regret_comparison',
                               directory='sources',
                               do_plot=False,
                               opt_difference=None):
    '''Plot multiple regret evolutions for comparison.'''

    i = 0
    plt.xlabel('Episodes')
    plt.ylabel('Regret')
    #plt.ylim(0, 150)

    if bethas is None:
        for regret in regrets:
            if errors is not None:
                plt.fill_between(np.cumsum(np.ones(c.EPOCHS)),
                                 regret + errors[i],
                                 regret - errors[i],
                                 label=plot_label,
                                 alpha=0.25)
                plt.plot(np.arange(c.EPOCHS),
                         regret + errors[i],
                         alpha=0.25,
                         ls='--')
                plt.plot(np.arange(c.EPOCHS),
                         regret - errors[i],
                         alpha=0.25,
                         ls='--')
                i += 1

            else:
                plt.plot(np.arange(c.EPOCHS),
                         regret,
                         label=plot_label)

    else:
        for regret in regrets:
            beta=bethas[i]
            if errors is not None:
                plt.fill_between(np.cumsum(np.ones(c.EPOCHS)),
                                 regret + errors[i],
                                 regret - errors[i],
                                 label=f"{plot_label}"+r" $\beta$ = "+f"{beta}",
                                 alpha=0.25)
                plt.plot(np.arange(c.EPOCHS),
                         regret + errors[i],
                         alpha=0.25,
                         ls='--')
                plt.plot(np.arange(c.EPOCHS),
                         regret - errors[i],
                         alpha=0.25,
                         ls='--')

            else:
                plt.plot(np.arange(c.EPOCHS),
                         regret,
                         label=f"{plot_label}"+r" $\beta$ = "+f"{beta}")

            i += 1

    if opt_difference is not None:
        plt.title(r'$||\theta^*-\theta_S||$='+f'{opt_difference}')

    plt.legend()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plotsuffix}.png')
        # plt.show()
        plt.close()


def multiple_beta_std_regret_plots(std_dev,
                                   bethas=None,
                                   plot_label=None,
                                   plotsuffix=None,
                                   directory='sources',
                                   do_plot=False,
                                   opt_difference=None):
    '''Plot standard deviation as a funcntion of time for different update rules.'''

    i = 0
    plt.xlabel('Episodes')
    plt.ylabel(r'$\sigma$(Regret)')
    plt.ylim(0, 175)

    if bethas is None:
        for std in std_dev:
            plt.plot(np.cumsum(np.ones(c.EPOCHS)),
                     std,
                     label=plot_label,
                     alpha=0.5)
            i += 1

    else:
        for std in std_dev:
            beta=bethas[i]
            plt.plot(np.cumsum(np.ones(c.EPOCHS)),
                     std,
                     label=f"{plot_label}"+r" $\beta$ = "+f"{beta}",
                     alpha=0.5)
            i += 1
    if opt_difference is not None:
        plt.title(r'$||\theta^*-\theta_S||$='+f'{opt_difference}')

    plt.legend()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plot_label}_{plotsuffix}regret_std_beta_comparison.png')
        # plt.show()
        plt.close()


def alpha_plots(alphas,
                betas=None,
                plot_label=None,
                do_plot=False,
                directory='',
                plotsuffix='alpha_comparison'):
    '''Plot evolution of alpha values for different update rules.'''

    plt.xlabel('Episode')
    plt.ylabel(r'$\mathrm{\alpha}$')

    i = 0

    if betas is None:
        for alpha in alphas:
            plt.plot(alpha, label=plot_label)

    else:
        for alpha in alphas:
            beta=betas[i]
            plt.plot(alpha, label=f"{plot_label}"+r" $\beta$ = "+f"{beta}")
            i += 1

    plt.legend()

    if do_plot:
        plt.savefig(f'{PLOT_DIR}{directory}/{plotsuffix}.png')
        # plt.show()
        plt.close()
