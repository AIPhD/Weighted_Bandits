import numpy as np
import matplotlib.pyplot as plt
import config as c
import real_data as rd
import training as t
import optimization as o

all_targets = rd.context_data_set
all_rewards = rd.reward_data_set

def real_weighted_training(target_data,
                           reward_data,
                           source=None,
                           estim=None,
                           alpha=0,
                           beta=c.BETA,
                           no_sources=1,
                           epochs=c.EPOCHS,
                           repeats=1,
                           update_rule='sigmoid',
                           arms_pulled_plot=False,
                           biased_reg=False,
                           exp_scale=1):
    '''Training algorithm based on weighted UCB function for linear bandits,
       adapted for real data.'''

    if source is None:
        source = np.zeros(len(target_data[0]))
        alpha = 0

    if estim is None:
        estim = np.abs(np.random.uniform(size=len(target_data[0])))


    if biased_reg:
        alpha = 0
        biased_alpha = 1/no_sources * np.ones((no_sources, repeats))

    dimension = len(target_data[0])
    context_size = len(target_data)
    theta_estim = estim/np.sqrt(np.dot(estim, estim))
    source_scale = np.ones((no_sources, repeats))
    theta_estim = np.tile(theta_estim, (repeats, 1))
    alpha_1 = alpha * np.ones((no_sources, repeats))
    alpha_2 = np.ones((repeats, 1)) - np.sum(alpha_1, axis=0)[:, np.newaxis]
    alpha_evol = np.zeros((1, epochs))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    gamma_s = c.GAMMA * np.ones((no_sources, repeats))
    no_pulls = np.zeros((repeats, context_size))
    rewards = np.zeros((repeats, epochs))

    try:
        a_matrix = c.LAMB * np.tile(np.identity(dimension), (repeats, 1, 1))
        a_inv = 1/c.LAMB * np.tile(np.identity(dimension), (repeats, 1, 1))
    except ZeroDivisionError:
        print('Lambda cannot be zero!')
        a_matrix = np.tile(np.identity(dimension), (repeats, 1, 1))
        a_inv = np.tile(np.identity(dimension), (repeats, 1, 1))

    b_vector = np.tile(np.zeros(dimension), (repeats, 1))
    regret_evol = np.zeros((repeats, epochs))
    x_history = []

    for i in range(0, epochs):
        index = np.argmax(t.ucb_weighted(target_data,
                                         a_inv,
                                         alpha_1,
                                         alpha_2,
                                         source,
                                         theta_estim,
                                         gamma_scalar,
                                         gamma_s,
                                         expl_scale=exp_scale), axis=1)

        instance = target_data[index]
        x_history.append(instance)
        r_real = reward_data[index]
        alpha_evol[:, i] = np.ndarray.flatten(np.asarray(alpha_2))
        rewards[:, i] = r_real
        source = np.einsum('mi,mij->mij', source_scale, source)
        y_s_new = np.einsum('lij,mij->mil', np.asarray(x_history), source)
        # source_scale = np.einsum('il,mil->mi',
        #                          rewards[:, :i+1],
        #                          y_s_new)/np.einsum('mil,mil->mi',
        #                                             y_s_new,
        #                                             y_s_new)
        # gamma_s = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_s_new[:, :, :i + 1])/
        #                             np.sqrt(np.einsum('lij,lij->li',
        #                                             np.asarray(x_history),
        #                                             np.asarray(x_history))).T,
        #                             axis=2)**2 + np.einsum('mij,mij->mi',
        #                                                 y_s_new - rewards[:, :i+1],
        #                                                 y_s_new - rewards[:, :i+1]))
        gamma_s = np.sqrt(2 + np.einsum('mij,mij->mi',
                                        y_s_new - rewards[:, :i+1],
                                        y_s_new - rewards[:, :i+1]))
        a_matrix += np.einsum('ij,ik->ijk', instance, instance)
        a_inv -= np.einsum('ijk,ikl->ijl',
                           a_inv,
                           np.einsum('ijk,ikl->ijl',
                                     np.einsum('ij,ik->ijk',
                                               instance,
                                               instance),
                                     a_inv))/(1 + np.einsum('ij,ij->i',
                                                            instance,
                                                            np.einsum('ijk,ik->ij',
                                                                      a_inv,
                                                                      instance)))[:,
                                                                                  np.newaxis,
                                                                                  np.newaxis]
        b_vector += r_real[:, np.newaxis] * instance
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)

        if biased_reg:
            weighted_source, biased_alpha = t.source_weighting(source,
                                                               rewards[:, :i+1],
                                                               y_s_new,
                                                               biased_alpha,
                                                               beta)
            theta_estim -= np.einsum('il,ijl->ij',
                                     weighted_source,
                                     np.einsum('ijk,ikl->ijl',
                                               a_inv,
                                               np.einsum('nik,nil->ikl',
                                                         np.asarray(x_history),
                                                         np.asarray(x_history))) -
                                     np.tile(np.identity(dimension), (repeats, 1, 1)))

        # gamma_t = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_t[:, :i + 1])/
        #                          np.sqrt(np.einsum('lij,lij->li',
        #                                            np.asarray(x_history),
        #                                            np.asarray(x_history))).T,
        #                          axis=1) + np.einsum('ij,ij->i',
        #                                              y_t - rewards[:, :i+1],
        #                                              y_t - rewards[:, :i+1]))[:, np.newaxis]
        gamma_scalar = np.asarray([np.sqrt(c.LAMB) +
                                   np.sqrt(2 * np.log(np.sqrt(np.linalg.det(a_matrix))/
                                                      (np.sqrt(c.LAMB**dimension) * c.DELTA)))]).T


        if update_rule=='hard':
            alpha_1, alpha_2 = t.hard_update_alphas(gamma_scalar,
                                                    gamma_s,
                                                    no_source=no_sources)

        elif update_rule=='sigmoid':
            alpha_1, alpha_2 = t.sigmoid_alpha_update(alpha_1,
                                                      alpha_2,
                                                      gamma_s,
                                                      gamma_scalar,
                                                      beta=beta)

        no_pulls[np.arange(repeats), index] += 1
        inst_regret = [5] - r_real
        regret_evol[:, i] = inst_regret
        regr=inst_regret
        print(f"Instant regret = {regr}")

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    mean_alpha = alpha_evol.sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)

    if arms_pulled_plot:
        plt.scatter(reward_data,
                    np.sum(no_pulls, axis=0)/repeats)
        plt.show()
        plt.close()

    return theta_estim, mean_regret, mean_alpha, regret_dev


def real_weighted_matrix_training(target_data,
                                  reward_data,
                                  source,
                                  estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                                  alpha=c.ALPHA,
                                  repeats=1,
                                  box_bounds=True,
                                  arms_pulled_plot=False):
    '''Training algorithm based on matrix weighted UCB function for linear bandits,
       adapted for real data.'''

    dimension = len(target_data[0])
    context_size = len(target_data)
    theta_estim = estim/np.sqrt(np.dot(estim, estim))
    source_scale = np.ones((repeats))
    theta_estim = np.tile(theta_estim, (repeats, 1))
    alpha_1 = alpha * np.tile(np.identity(dimension), (repeats, 1, 1))
    alpha_2 = np.tile(np.identity(dimension), (repeats, 1, 1)) - alpha_1
    alpha_evol = np.zeros((repeats, c.EPOCHS))
    gamma_t = np.tile(c.GAMMA, (repeats, 1))
    gamma_s = c.GAMMA * np.ones((repeats))
    no_pulls = np.zeros((repeats, context_size))
    rewards = np.zeros((repeats, c.EPOCHS))

    try:
        a_matrix = c.LAMB * np.tile(np.identity(dimension), (repeats, 1, 1))
        a_inv = 1/c.LAMB * np.tile(np.identity(dimension), (repeats, 1, 1))
    except ZeroDivisionError:
        print('Lambda cannot be zero!')
        a_matrix = np.tile(np.identity(dimension), (repeats, 1, 1))
        a_inv = np.tile(np.identity(dimension), (repeats, 1, 1))

    b_vector = np.tile(np.zeros(dimension), (repeats, 1))
    regret_evol = np.zeros((repeats, c.EPOCHS))
    x_history = []

    for i in range(0, c.EPOCHS):
        source = np.einsum('i,ij->ij', source_scale, source)
        index = np.argmax(t.ucb_matrix_weighted(target_data,
                                                a_inv,
                                                alpha_1,
                                                alpha_2,
                                                source,
                                                theta_estim,
                                                gamma_t,
                                                gamma_s), axis=1)

        instance = target_data[index]
        x_history.append(instance)
        r_real = reward_data[index]
        alpha_evol[:, i] = np.ndarray.flatten(np.asarray(np.sum(np.sum(alpha_1, axis=1), axis=1)))
        rewards[:, i] = r_real
        y_t = np.einsum('lij,ij->il', np.asarray(x_history), theta_estim)
        y_s_new = np.einsum('lij,ij->il', np.asarray(x_history), source)
        # source_scale = np.einsum('il,il->i',
        #                          rewards[:, :i+1],
        #                          y_s_new)/np.einsum('il,il->i',
        #                                             y_s_new,
        #                                             y_s_new)
        gamma_s = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_s_new[:, :i + 1])/
                                 np.sqrt(np.einsum('lij,lij->li',
                                                   np.asarray(x_history),
                                                   np.asarray(x_history))).T,
                                 axis=1)**2 + np.einsum('ij,ij->i',
                                                        y_s_new - rewards[:, :i+1],
                                                        y_s_new - rewards[:, :i+1]))

        # gamma_s = np.sqrt(1 + np.einsum('mij,mij->mi',
        #                                 y_s_new - rewards[:, :i+1],
        #                                 y_s_new - rewards[:, :i+1]))

        gamma_t = np.sqrt(np.max(np.abs(rewards[:, :i + 1] - y_t[:, :i + 1])/
                                 np.sqrt(np.einsum('lij,lij->li',
                                                   np.asarray(x_history),
                                                   np.asarray(x_history))).T,
                                 axis=1)**2 + np.einsum('ij,ij->i',
                                                        y_t - rewards[:, :i+1],
                                                        y_t - rewards[:, :i+1]))[:, np.newaxis]

        a_matrix += np.einsum('ij,ik->ijk', instance, instance)
        a_inv -= np.einsum('ijk,ikl->ijl',
                           a_inv,
                           np.einsum('ijk,ikl->ijl',
                                     np.einsum('ij,ik->ijk',
                                               instance,
                                               instance),
                                     a_inv))/(1 + np.einsum('ij,ij->i',
                                                            instance,
                                                            np.einsum('ijk,ik->ij',
                                                                      a_inv,
                                                                      instance)))[:,
                                                                                  np.newaxis,
                                                                                  np.newaxis]
        b_vector += r_real[:, np.newaxis] * instance
        theta_estim = np.einsum('ijk,ik->ij', a_inv, b_vector)


        if box_bounds:
            alpha_1, alpha_2 = o.least_squares_bounded_opt(np.asarray(x_history),
                                                           rewards[:, :i+1],
                                                           source,
                                                           theta_estim)
        else:
            alpha_1, alpha_2 = o.least_squares_opt(np.asarray(x_history),
                                                   rewards[:, :i+1],
                                                   source,
                                                   theta_estim)

        no_pulls[np.arange(repeats), index] += 1
        inst_regret = 5 - r_real
        regret_evol[:, i] = inst_regret
        regr = inst_regret
        print(f"Instant regret = {regr}")

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    mean_alpha = alpha_evol.sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)

    if arms_pulled_plot:
        plt.scatter(reward_data,
                    np.sum(no_pulls, axis=0)/repeats)
        plt.show()
        plt.close()

    return theta_estim, mean_regret, mean_alpha, regret_dev


# all_targets = rd.context_data_set
# all_rewards = rd.reward_data_set
# for j in range(0, len(all_rewards)):
#     target_data_set, reward_data_set = rd.extract_context_for_users(j, all_targets, all_rewards)
#     theta_new_s, regret, alphas_evo, regret_std = real_weighted_training(target_data_set,
#                                                                          reward_data_set)
#     np.save(f'source_bandits/bandit_{j}', theta_new_s)
