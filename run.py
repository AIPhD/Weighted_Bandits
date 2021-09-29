import training as t
import evaluation as e


def main():
    '''Main function used to evaluate different weighted bandit settings.'''

    betas = [0.1, 1]

    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(alpha=0,
                                                                         update_rule='sigmoid')
    regret_hard, alpha_hard, regret_std_hard = t.weighted_training()
    # regret_soft, alpha_soft, regret_std_soft = t.compared_betas(betas,
    #                                                             update_rule='softmax')
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
                                 plot_label='sigmoid',
                                 do_plot=True)

    e.multiple_beta_regret_over_time_plots([regret_linucb],
                                           [regret_std_linucb],
                                           plot_label='linUCB')
    e.multiple_beta_regret_over_time_plots([regret_hard],
                                           [regret_std_hard],
                                           plot_label=r'Hard $\alpha$ update rule')
    # e.multiple_beta_regret_over_time_plots(regret_soft,
    #                                        regret_std_soft,
    #                                        betas,
    #                                        plot_label='softmax')
    e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                           regret_std_sigmoid,
                                           betas,
                                           plot_label='sigmoid',
                                           do_plot=True)

    e.alpha_plots([alpha_linucb], plot_label='linUCB')
    e.alpha_plots([alpha_hard], plot_label=r'Hard $\alpha$ update rule')
    # e.alpha_plots(alpha_soft, betas, plot_label='softmax')
    e.alpha_plots(alpha_sigmoid, betas, do_plot=True, plot_label='sigmoid')


if __name__ == '__main__':
    main()
