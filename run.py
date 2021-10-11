import numpy as np
import synthetic_data as sd
import config as c
import training as t
import evaluation as e


def main():
    '''Main function used to evaluate different weighted bandit settings.'''

    target_class = sd.TargetContext()
    estim = np.abs(np.random.uniform(size=c.DIMENSION))

    #source_comparison(target_class, estim)
    for i in range(14, c.DIMENSION_ALIGN):
        source = target_class.source_bandits(c.NO_SOURCES, dim_align=i)
        align_comparison(target_class, source, estim, i, probalistic=True)
        align_comparison(target_class, source, estim, i)


def align_comparison(target_class,
                     source,
                     estim,
                     dim_align,
                     soft_comparison=False,
                     probalistic=False):
    '''Script to compare single source weighted bandits, with different bandits'''

    i = dim_align

    if probalistic:
        prob_string='probalistic_'

    else:
        prob_string = ''

    betas = [0.01, 0.1]
    opt = target_class.theta_opt
    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(target=target_class,
                                                                         source=source,
                                                                         theta_estim=estim,
                                                                         alpha=0,
                                                                         update_rule='soft',
                                                                         probalistic=probalistic)
    regret_hard, alpha_hard, regret_std_hard = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   theta_estim=estim,
                                                                   probalistic=probalistic)

    if soft_comparison:
        regret_soft, alpha_soft, regret_std_soft = t.compared_betas(target_class,
                                                                    source,
                                                                    betas,
                                                                    theta_estim=estim,
                                                                    update_rule='softmax',
                                                                    probalistic=probalistic)

    regret_sigmoid, alpha_sigmoid, regret_std_sigmoid = t.compared_betas(target_class,
                                                                         source,
                                                                         betas,
                                                                         theta_estim=estim,
                                                                         update_rule='sigmoid',
                                                                         probalistic=probalistic)
    e.multiple_beta_regret_plots([regret_linucb],
                                 [regret_std_linucb],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([regret_hard],
                                 [regret_std_hard],
                                 plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_plots(regret_sigmoid,
                                 regret_std_sigmoid,
                                 betas,
                                 plot_label='sigmoid',
                                 plotsuffix=f'{i}_align_{prob_string}regret_beta_comparison',
                                 dir='dim_align',
                                 do_plot=True,
                                 opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                         opt-source[:, 0, :],
                                                                         opt-source[:, 0, :]))))

    e.multiple_beta_std_regret_plots([regret_std_linucb],
                                     plot_label='linUCB',
                                     opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :],
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :]))))
    e.multiple_beta_std_regret_plots([regret_std_hard],
                                     plot_label='hard',
                                     opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :],
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :]))))
    e.multiple_beta_std_regret_plots(regret_std_sigmoid,
                                     betas,
                                     plot_label='sigmoid',
                                     plotsuffix=f'{i}_align_{prob_string}',
                                     dir='dim_align',
                                     do_plot=True,
                                     opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :],
                                                                             opt-source[:,
                                                                                        0,
                                                                                        :]))))


    e.multiple_beta_regret_over_time_plots([regret_linucb],
                                           [regret_std_linucb],
                                           plot_label='linUCB')
    e.multiple_beta_regret_over_time_plots([regret_hard],
                                           [regret_std_hard],
                                           plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                           regret_std_sigmoid,
                                           betas,
                                           plot_label='sigmoid',
                                           plotsuffix=f'{i}_align_{prob_string}',
                                           do_plot=True)

    if soft_comparison:
        e.multiple_beta_regret_plots([regret_linucb],
                                     [regret_std_linucb],
                                     plot_label='linUCB')
        e.multiple_beta_regret_plots(regret_soft,
                                     regret_std_soft,
                                     betas,
                                     plot_label='softmax',
                                     do_plot=True)
        e.multiple_beta_regret_over_time_plots([regret_linucb],
                                               [regret_std_linucb],
                                               plot_label='linUCB')
        e.multiple_beta_regret_over_time_plots(regret_soft,
                                               regret_std_soft,
                                               betas,
                                               plot_label='softmax',
                                               do_plot=True)
        e.alpha_plots(alpha_soft, betas, do_plot=True, plot_label='softmax')

    e.alpha_plots([alpha_linucb], plot_label='linUCB')
    e.alpha_plots([alpha_hard], plot_label=r'Hard $\alpha$ update rule')
    e.alpha_plots(alpha_sigmoid,
                  betas,
                  do_plot=True,
                  plot_label='sigmoid',
                  plotsuffix=f'{i}_align_{prob_string}alpha_comparison')

def source_comparison(target_class, estim, probalistic=False):
    '''Script to compare weighted bandits with different update strategies and ammout of sources'''

    sources=[]

    if probalistic:
        prob_string = 'probalistic_'
    else:
        prob_string = ''

    for i in range(1, c.NO_SOURCES + 1):
        sources.append(target_class.source_bandit())
        source = np.asarray(sources)
        betas = [0.01, 0.1, 1]
        opt = target_class.theta_opt
        regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(target=target_class,
                                                                             source=source,
                                                                             theta_estim=estim,
                                                                             no_sources=i,
                                                                             alpha=0,
                                                                             update_rule='soft',
                                                                             probalistic=probalistic)
        regret_hard, alpha_hard, regret_std_hard = t.weighted_training(target=target_class,
                                                                    source=source,
                                                                    theta_estim=estim,
                                                                    no_sources=i,
                                                                    alpha=1/(i+1),
                                                                    probalistic=probalistic)
        # regret_soft, alpha_soft, regret_std_soft = t.compared_betas(target_class,
        #                                                             source,
        #                                                             betas,
        #                                                             theta_estim=estim,
        #                                                             update_rule='softmax',
        #                                                             probalistic=probalistic)
        regret_sigmoid, alpha_sigmoid, regret_std_sigmoid = t.compared_betas(target_class,
                                                                            source,
                                                                            betas,
                                                                            theta_estim=estim,
                                                                            update_rule='sigmoid',
                                                                            probalistic=probalistic)
        e.multiple_beta_regret_plots([regret_linucb],
                                     [regret_std_linucb],
                                     plot_label='linUCB')
        e.multiple_beta_regret_plots([regret_hard],
                                     [regret_std_hard],
                                     plot_label=r'Hard $\alpha$ update rule')
        e.multiple_beta_regret_plots(regret_sigmoid,
                                     regret_std_sigmoid,
                                     betas,
                                     plot_label='sigmoid',
                                     plotsuffix=f'{i}_sources_{prob_string}regret_beta_comparison',
                                     do_plot=True,
                                     opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                             opt-source[:, 0, :],
                                                                             opt-source[:, 0, :]))))

        e.multiple_beta_std_regret_plots([regret_std_linucb],
                                         betas,
                                         plot_label='linUCB',
                                         opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :],
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :]))))
        e.multiple_beta_std_regret_plots([regret_std_hard],
                                         betas,
                                         plot_label='hard',
                                         opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :],
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :]))))
        e.multiple_beta_std_regret_plots(regret_std_sigmoid,
                                         betas,
                                         plot_label='sigmoid',
                                         plotsuffix=f'{i}_sources_{prob_string}',
                                         do_plot=True,
                                         opt_difference=np.min(np.sqrt(np.einsum('mj,mj->m',
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :],
                                                                                 opt-source[:,
                                                                                            0,
                                                                                            :]))))
        # e.multiple_beta_regret_plots([regret_linucb],
        #                              [regret_std_linucb],
        #                              plot_label='linUCB')
        # e.multiple_beta_regret_plots(regret_soft,
        #                              regret_std_soft,
        #                              betas,
        #                              plot_label='softmax',
        #                              do_plot=True)


        e.multiple_beta_regret_over_time_plots([regret_linucb],
                                               [regret_std_linucb],
                                               plot_label='linUCB')
        e.multiple_beta_regret_over_time_plots([regret_hard],
                                               [regret_std_hard],
                                               plot_label=r'Hard $\alpha$ update rule')
        e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                               regret_std_sigmoid,
                                               betas,
                                               plot_label='sigmoid',
                                               plotsuffix=f'{i}_sources_{prob_string}',
                                               do_plot=True)
        # e.multiple_beta_regret_over_time_plots([regret_linucb],
        #                                        [regret_std_linucb],
        #                                        plot_label='linUCB')
        # e.multiple_beta_regret_over_time_plots(regret_soft,
        #                                        regret_std_soft,
        #                                        betas,
        #                                        plot_label='softmax',
        #                                        do_plot=True)

        e.alpha_plots([alpha_linucb], plot_label='linUCB')
        # e.alpha_plots(alpha_soft, betas, do_plot=True, plot_label='softmax')
        e.alpha_plots([alpha_hard], plot_label=r'Hard $\alpha$ update rule')
        e.alpha_plots(alpha_sigmoid,
                      betas,
                      do_plot=True,
                      plot_label='sigmoid',
                      plotsuffix=f'{i}_sources_{prob_string}alpha_comparison')


if __name__ == '__main__':
    main()
