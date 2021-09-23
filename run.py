import matplotlib
import matplotlib.pyplot as plt
import training as t
import evaluation as e
matplotlib.use('TKAgg')


def main():
    '''Main function used to evaluate different weighted bandit settings.'''
    betas = [0.1]

    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(alpha=0,
                                                                         update_rule='softmax')
    regret_hard, alpha_hard, regret_std_hard = t.weighted_training()
    regret_soft, alpha_soft, regret_std_soft = t.compared_betas(betas,
                                                                update_rule='softmax')
    regret_sigmoid, alpha_sigmoid, regret_std_sigmoid = t.compared_betas(betas,
                                                                         update_rule='sigmoid')

    e.multiple_beta_regret_plots([regret_linucb],
                                 [regret_std_linucb],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([regret_hard],
                                 [regret_std_hard],
                                 plot_label=r'Hard $\alpha$ update rule')
    # e.multiple_beta_regret_plots(regret_soft,
    #                              regret_std_soft,
    #                              betas,
    #                              plot_label='softmax')
    e.multiple_beta_regret_plots(regret_sigmoid,
                                 regret_std_sigmoid,
                                 betas,
                                 plot_label='sigmoid')
    plt.show()
    plt.close()

    e.multiple_beta_regret_over_time_plots([regret_linucb],
                                           [regret_std_linucb],
                                           plot_label='linUCB')
    e.multiple_beta_regret_over_time_plots([regret_hard],
                                           [regret_std_hard],
                                           plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_over_time_plots(regret_soft,
                                           regret_std_soft,
                                           betas,
                                           plot_label='softmax')
    e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                           regret_std_sigmoid,
                                           betas,
                                           plot_label='sigmoid')

    plt.show()
    plt.savefig('/home/steven/weighted_bandits/plots/regret_comparison.png')
    plt.close()
    e.alpha_plots([alpha_hard])
    e.alpha_plots(alpha_sigmoid, betas)


if __name__ == '__main__':
    main()
