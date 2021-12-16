import numpy as np
import synthetic_data as sd
import config as c
import training as t
import evaluation as e
import real_data as rd
import real_data_training as rdt


def main():
    '''Main function used to evaluate different weighted bandit settings.'''

    filter_data = [
                #    ['F', 18, '0', True],
                #    ['F', 18, '0', False],
                #    ['F', 35, '12', True],
                #    ['F', 35, '12', False],
                #    ['F', 25, '4', False],
                #    ['F', 25, '4', True],
                #    ['F', 35, '6', False],
                #    ['F', 35, '6', True],
                #    ['F', 35, '9', False],
                #    ['F', 35, '9', True],
                #    ['F', 35, '11', True],
                #    ['F', 35, '11', False],
                #    ['F', 35, '19', False],
                #    ['F', 35, '19', True],
                #    ['F', 45, '15', True],
                #    ['F', 45, '15', False],
                #    ['M', 18, '0', True],
                   ['M', 18, '0', False],
                #    ['M', 35, '2', False],
                #    ['M', 35, '2', True],
                #    ['M', 35, '3', False],
                #    ['M', 35, '3', True],
                #    ['M', 35, '15', False],
                #    ['M', 35, '15', True],
                #    ['M', 35, '17', False],
                #    ['M', 35, '17', True],
                #    ['M', 45, '3', True],
                #    ['M', 45, '3', False],
                #    ['M', 45, '20', False],
                   ['M', 45, '20', True],
                #    ['M', 56, '13', True],
                #    ['M', 56, '13', False]
                   ]

    for filt in filter_data:
        real_data_comparison(gender=filt[0], age=filt[1], prof=filt[2], multi_source=filt[3])

    # estim = np.abs(np.random.uniform(size=c.DIMENSION))
    # target_class = sd.TargetContext()
    # sources=[]

    # for i in range(1, c.NO_SOURCES + 1):
    #     sources.append(target_class.source_bandit())
    #     source = np.asarray(sources)
    #     conf = np.einsum('mij,mij->mi',
    #                      target_class.theta_opt-source,
    #                      target_class.theta_opt-source)
    #     source_comparison(target_class, estim, source, i,theta_conf=conf, probalistic=True)
    #     source_comparison(target_class, estim, source, i)

    # for i in range(30, c.DIMENSION_ALIGN + 1):
    #     source = target_class.source_bandits(c.NO_SOURCES, dim_align=i)
    #     align_comparison(target_class, source, estim, i, probalistic=True, soft_comparison=False)
    #     align_comparison(target_class, source, estim, i, soft_comparison=False)

    # source = target_class.source_bandits(c.NO_SOURCES, dim_align=20)
    # matrix_comparison(target_class, source, estim)


def real_data_comparison(gender=None,
                         age=None,
                         prof=None,
                         context=rd.context_data_set,
                         rewards=rd.reward_data_set,
                         multi_source=False):
    '''Compares multiple algorithms on real data'''

    real_sources = []
    dimension = len(context[0])
    filtered_users = rd.filter_users(np.asarray(rd.user_data_set),
                                     gender=gender,
                                     age=age,
                                     prof=prof)
    filtered_user_index = np.asarray([int(i) for i in filtered_users[:, 0]]) - 1
    train_user_ind = np.random.randint(0, len(filtered_users[:, 0]))
    real_target, real_rewards = rd.extract_context_for_users(filtered_user_index[train_user_ind],
                                                             context,
                                                             rewards)

    if multi_source:
        source_indices = np.delete(filtered_user_index, train_user_ind)
        mult_str = '_multi'

    else:
        source_user_ind = np.random.randint(0, len(filtered_users[:, 0]) - 1)
        source_indices = [np.delete(filtered_user_index, train_user_ind)[source_user_ind]]
        mult_str = ''

    estim = np.abs(np.random.uniform(size=len(real_target[0])))

    for i in source_indices:
        real_sources.append(np.load(f'source_bandits/bandit_{i}.npy'))

    real_source = np.asarray(real_sources)

    linucb_output = rdt.real_weighted_training(real_target,
                                               real_rewards,
                                               source=real_source,
                                               estim=estim,
                                               alpha=0,
                                               no_sources=len(source_indices),
                                               update_rule='sigmoid',
                                               exp_scale=1)

    hard_output = rdt.real_weighted_training(real_target,
                                             real_rewards,
                                             source=real_source,
                                             estim=estim,
                                             alpha=1/(len(source_indices)),
                                             no_sources=len(source_indices),
                                             update_rule='hard',
                                             exp_scale=1)

    biased_output = rdt.real_weighted_training(real_target,
                                               real_rewards,
                                               source=real_source,
                                               estim=estim,
                                               biased_reg=True,
                                               no_sources=len(source_indices),
                                               update_rule=None,
                                               exp_scale=1)

    if not multi_source:
        matrix_output = rdt.real_weighted_matrix_training(real_target,
                                                          real_rewards,
                                                          source=real_source[0],
                                                          estim=estim,
                                                          alpha=1/(len(source_indices)+1))

    sigmoid_output = rdt.real_weighted_training(real_target,
                                                real_rewards,
                                                source=real_source,
                                                estim=estim,
                                                alpha=1/(len(source_indices)+1),
                                                no_sources=len(source_indices),
                                                update_rule='sigmoid',
                                                exp_scale=1)

    e.multiple_beta_regret_plots([linucb_output[1]],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([sigmoid_output[1]],
                                 bethas=[0.1],
                                 plot_label='sigmoid')
    e.multiple_beta_regret_plots([biased_output[1]],
                                 bethas=[0.1],
                                 plot_label='biased regularization')

    if not multi_source:
        e.multiple_beta_regret_plots([matrix_output[1]],
                                     plot_label='matrix weighted')

    e.multiple_beta_regret_plots([hard_output[1]],
                                 directory='real_dim_align_comparison',
                                 plot_label='hard',
                                 plotsuffix=f'real_regret_comparison_{gender}_{age}_{prof}{mult_str}',
                                 do_plot=True)

    e.alpha_plots([dimension * linucb_output[2]],
                  plot_label='linUCB')
    e.alpha_plots([dimension * hard_output[2]],
                  plot_label='hard')
    e.alpha_plots([dimension * biased_output[2]],
                  plot_label='biased Regularization')

    if not multi_source:
        e.alpha_plots([matrix_output[2]],
                      plot_label='matrix')

    e.alpha_plots([dimension * sigmoid_output[2]],
                  plot_label='sigmoid',
                  do_plot=True,
                  directory='alpha_comparison',
                  plotsuffix=f'real_weight_comparison_{gender}_{age}_{prof}{mult_str}')


def matrix_comparison(target_class,
                      source,
                      estim):
    '''Script to compare single source matrix weighted bandits, with different bandits'''

    opt = target_class.theta_opt
    dimension = len(opt)
    theta_diff = np.sqrt(np.dot(opt-source[0][0], opt-source[0][0]))
    regret_matrix, alpha_matrix, regret_std_matrix = t.weighted_matrix_training(target=target_class,
                                                                                source=source[0],
                                                                                estim=estim,
                                                                                box_bounds=True)
    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(target=target_class,
                                                                         source=source,
                                                                         estim=estim,
                                                                         alpha=0,
                                                                         update_rule=None)
    regret_bias, alpha_bias, regret_std_bias = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   estim=estim,
                                                                   alpha=0,
                                                                   update_rule=None,
                                                                   biased_reg=True)
    regret_hard, alpha_hard, regret_std_hard = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   estim=estim,
                                                                   exp_scale=0.1)


    e.multiple_beta_regret_plots([regret_linucb],
                                 [regret_std_linucb],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([regret_hard],
                                 [regret_std_hard],
                                 plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_plots([regret_bias],
                                 [regret_std_bias],
                                 plot_label='Biased Regularization')
    e.multiple_beta_regret_plots([regret_matrix],
                                 [regret_std_matrix],
                                 plot_label='matrix weighted',
                                 directory='matrix_comparison',
                                 plotsuffix='regret_matrix_comparison',
                                 do_plot=True,
                                 opt_difference=np.round(theta_diff, 3))

    e.multiple_beta_std_regret_plots([regret_std_linucb],
                                     plot_label='linUCB')
    e.multiple_beta_std_regret_plots([regret_std_hard],
                                     plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_std_regret_plots([regret_std_bias],
                                     plot_label='Biased regularization')
    e.multiple_beta_std_regret_plots([regret_std_matrix],
                                     plot_label='matrix weighted',
                                     directory='matrix_comparison/std',
                                     do_plot=True,
                                     opt_difference=np.round(theta_diff, 3))

    e.alpha_plots(dimension * [alpha_linucb],
                  plot_label='linUCB')
    e.alpha_plots(dimension * [alpha_hard],
                  plot_label='hard')
    e.alpha_plots(dimension * [alpha_bias], plot_label='Biased Regularization')
    e.alpha_plots([alpha_matrix],
                  do_plot=True,
                  plot_label='matrix weighted',
                  plotsuffix='alpha_matrix_comparison',
                  directory='alpha_comparison')


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

    betas = [0.1, 1]
    opt = target_class.theta_opt
    theta_diff = np.min(np.sqrt(np.einsum('mj,mj->m', opt-source[:, 0, :], opt-source[:, 0, :])))
    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(target=target_class,
                                                                         source=source,
                                                                         estim=estim,
                                                                         alpha=0,
                                                                         update_rule=None,
                                                                         probalistic=probalistic,
                                                                         exp_scale=0.1)

    regret_bias, alpha_bias, regret_std_bias = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   estim=estim,
                                                                   alpha=0,
                                                                   update_rule=None,
                                                                   probalistic=probalistic,
                                                                   biased_reg=True,
                                                                   exp_scale=0.1)

    regret_hard, alpha_hard, regret_std_hard = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   estim=estim,
                                                                   probalistic=probalistic,
                                                                   exp_scale=0.1)

    if soft_comparison:
        regret_soft, alpha_soft, regret_std_soft = t.compared_betas(target_class,
                                                                    source,
                                                                    betas,
                                                                    estim=estim,
                                                                    update_rule='softmax',
                                                                    probalistic=probalistic)

    regret_sigmoid, alpha_sigmoid, regret_std_sigmoid = t.compared_betas(target_class,
                                                                         source,
                                                                         betas,
                                                                         estim=estim,
                                                                         update_rule='sigmoid',
                                                                         probalistic=probalistic,
                                                                         exp_scale=0.1)

    e.multiple_beta_regret_plots([regret_linucb],
                                 [regret_std_linucb],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([regret_hard],
                                 [regret_std_hard],
                                 plot_label=r'Hard $\alpha$ update rule')

    if soft_comparison:
        e.multiple_beta_regret_plots(regret_soft,
                                     regret_std_soft,
                                     betas,
                                     directory='dim_align_comparison',
                                     plot_label='softmax')

    e.multiple_beta_regret_plots([regret_bias],
                                 [regret_std_bias],
                                 plot_label='Biased Regularization')
    e.multiple_beta_regret_plots(regret_sigmoid,
                                 regret_std_sigmoid,
                                 betas,
                                 plot_label='sigmoid',
                                 plotsuffix=f'{i}_align_{prob_string}regret_beta_comparison',
                                 directory='dim_align_comparison',
                                 do_plot=True,
                                 opt_difference=np.round(theta_diff, 3))

    e.multiple_beta_std_regret_plots([regret_std_linucb],
                                     plot_label='linUCB')
    e.multiple_beta_std_regret_plots([regret_std_hard],
                                     plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_std_regret_plots([regret_std_bias],
                                     plot_label='Biased regularization')

    if soft_comparison:
        e.multiple_beta_regret_over_time_plots(regret_soft,
                                               regret_std_soft,
                                               betas,
                                               directory='dim_align_comparison/regret_over_time',
                                               plot_label='softmax')

    e.multiple_beta_std_regret_plots(regret_std_sigmoid,
                                     betas,
                                     plot_label='sigmoid',
                                     plotsuffix=f'{i}_align_{prob_string}',
                                     directory='dim_align_comparison/std',
                                     do_plot=True,
                                     opt_difference=np.round(theta_diff, 3))

    e.multiple_beta_regret_over_time_plots([regret_linucb],
                                           [regret_std_linucb],
                                           plot_label='linUCB')
    e.multiple_beta_regret_over_time_plots([regret_hard],
                                           [regret_std_hard],
                                           plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_over_time_plots([regret_bias],
                                           [regret_std_bias],
                                           plot_label='Biased regularization')
    e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                           regret_std_sigmoid,
                                           betas,
                                           plot_label='sigmoid',
                                           directory='dim_align_comparison/regret_over_time',
                                           plotsuffix=f'{i}_align_{prob_string}',
                                           do_plot=True)


    if soft_comparison:
        e.alpha_plots(alpha_soft,
                      betas,
                      directory='alpha_comparison',
                      plot_label='softmax')
    e.alpha_plots([alpha_linucb], plot_label='linUCB')
    e.alpha_plots([alpha_hard], plot_label=r'Hard $\alpha$ update rule')
    e.alpha_plots([alpha_bias], plot_label='Biased Regularization')
    e.alpha_plots(alpha_sigmoid,
                  betas,
                  do_plot=True,
                  plot_label='sigmoid',
                  directory='alpha_comparison',
                  plotsuffix=f'{i}_align_{prob_string}alpha_comparison')


def source_comparison(target_class,
                      estim,
                      source,
                      no_sources,
                      theta_conf=None,
                      regret_over_time=False,
                      probalistic=False):
    '''Script to compare weighted bandits with different update strategies and ammout of sources'''

    i = no_sources

    if probalistic:
        prob_string = 'probalistic_'
    else:
        prob_string = ''

    betas = [0.1]
    opt = target_class.theta_opt
    theta_diff = np.min(np.sqrt(np.einsum('mj,mj->m', opt-source[:, 0, :], opt-source[:, 0, :])))
    regret_linucb, alpha_linucb, regret_std_linucb = t.weighted_training(target=target_class,
                                                                         source=source,
                                                                         estim=estim,
                                                                         no_sources=i,
                                                                         alpha=0,
                                                                         theta_conf=theta_conf,
                                                                         update_rule=None,
                                                                         exp_scale=0.1,
                                                                         probalistic=probalistic)
    regret_hard, alpha_hard, regret_std_hard = t.weighted_training(target=target_class,
                                                                   source=source,
                                                                   estim=estim,
                                                                   no_sources=i,
                                                                   alpha=1/(i+1),
                                                                   exp_scale=0.1,
                                                                   probalistic=probalistic)
    # regret_soft, alpha_soft, regret_std_soft = t.compared_betas(target_class,
    #                                                             source,
    #                                                             betas,
    #                                                             estim=estim,
    #                                                             update_rule='softmax',
    #                                                             probalistic=probalistic)
    regret_sigmoid, alpha_sigmoid, regret_std_sigmoid = t.compared_betas(target_class,
                                                                         source,
                                                                         betas,
                                                                         estim=estim,
                                                                         exp_scale=0.1,
                                                                         update_rule='sigmoid',
                                                                         probalistic=probalistic)
    regret_biased, alpha_biased, regret_std_biased = t.compared_betas(target_class,
                                                                      source,
                                                                      betas,
                                                                      estim=estim,
                                                                      biased_reg=True,
                                                                      exp_scale=0.1,
                                                                      update_rule=None,
                                                                      probalistic=probalistic)
    e.multiple_beta_regret_plots([regret_linucb],
                                 [regret_std_linucb],
                                 plot_label='linUCB')
    e.multiple_beta_regret_plots([regret_hard],
                                 [regret_std_hard],
                                 plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_plots(regret_biased,
                                 regret_std_biased,
                                 betas,
                                 plot_label='Biased regularization')
    e.multiple_beta_regret_plots(regret_sigmoid,
                                 regret_std_sigmoid,
                                 betas,
                                 plot_label='sigmoid',
                                 plotsuffix=f'{i}_sources_{prob_string}regret_beta_comparison',
                                 do_plot=True,
                                 directory='sources_comparison',
                                 opt_difference=np.round(theta_diff, 3))

    e.multiple_beta_std_regret_plots([regret_std_linucb],
                                     plot_label='linUCB')
    e.multiple_beta_std_regret_plots([regret_std_hard],
                                     plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_std_regret_plots(regret_std_biased,
                                     betas,
                                     plot_label='Biased regularization')
    e.multiple_beta_std_regret_plots(regret_std_sigmoid,
                                     betas,
                                     plot_label='sigmoid',
                                     plotsuffix=f'{i}_sources_{prob_string}',
                                     do_plot=True,
                                     directory='sources_comparison/std',
                                     opt_difference=np.round(theta_diff, 3))
    # e.multiple_beta_regret_plots([regret_linucb],
    #                              [regret_std_linucb],
    #                              plot_label='linUCB')
    # e.multiple_beta_regret_plots(regret_soft,
    #                              regret_std_soft,
    #                              betas,
    #                              plot_label='softmax',
    #                              do_plot=True)


    if regret_over_time:
        e.multiple_beta_regret_over_time_plots([regret_linucb],
                                               [regret_std_linucb],
                                               plot_label='linUCB')
        e.multiple_beta_regret_over_time_plots([regret_hard],
                                               [regret_std_hard],
                                               plot_label=r'Hard $\alpha$ update rule')
        e.multiple_beta_regret_over_time_plots(regret_biased,
                                               regret_std_biased,
                                               betas,
                                               plot_label='Biased regularization')
        e.multiple_beta_regret_over_time_plots(regret_sigmoid,
                                               regret_std_sigmoid,
                                               betas,
                                               plot_label='sigmoid',
                                               plotsuffix=f'{i}_sources_{prob_string}',
                                               directory='sources_comparison/regret_over_time',
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
    e.alpha_plots(alpha_biased, betas, plot_label='Biased regularization')
    e.alpha_plots(alpha_sigmoid,
                  betas,
                  do_plot=True,
                  directory='alpha_comparison',
                  plot_label='sigmoid',
                  plotsuffix=f'{i}_sources_{prob_string}alpha_comparison')


if __name__ == '__main__':
    main()
