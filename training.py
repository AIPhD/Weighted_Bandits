# from scipy import special as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import synthetic_data as sd
import config as c
matplotlib.use('TKAgg')


def ucb_weighted(x_instances, a_inv, alpha_1, alpha_2, theta_s, theta_t, gamma_t, gamma_s=0):
    '''UCB function based on weighted contributions of both domains.'''

    reward_estim_s = np.einsum('mi,mij->ij', alpha_1, np.einsum('jk,mik->mij',
                                                                x_instances,
                                                                theta_s))
    reward_estim_t = alpha_2 * np.einsum('jk,ik->ij', x_instances, theta_t)
    gamma = np.einsum('mi,mi->i', alpha_1, gamma_s)[:, np.newaxis] + alpha_2 * gamma_t
    expl = 0.1 * gamma * np.sqrt(np.einsum('jk,ikj->ij',
                                           x_instances,
                                           np.einsum('ijk,lk->ijl',
                                                     a_inv,
                                                     x_instances)))
    return reward_estim_s + reward_estim_t + expl


def probalistic_ucb(x_instances, a_inv, alpha_1, theta_s, theta_t, gamma_t, gamma_s=0):
    '''UCB function based on one single bandit chosed randomly with probability given by alpha'''

    prob = np.random.uniform(size=c.REPEATS)
    prob_limits = np.append(np.cumsum(alpha_1, axis=0), np.ones((1, c.REPEATS)), axis=0)
    prob_index_list = prob_limits - prob
    ind_list = np.argmin(np.where(prob_index_list >= 0, prob_index_list, 1), axis=0)
    theta_s = np.append(theta_s, np.asarray([theta_t]), axis=0)
    gamma_s = np.append(gamma_s, gamma_t.T, axis=0)
    reward_estim = np.einsum('jk,ik->ij', x_instances, theta_s[ind_list, np.arange(c.REPEATS)])
    gamma = gamma_s[ind_list, np.arange(c.REPEATS)][:, np.newaxis]
    expl = 0.1 * gamma * np.sqrt(np.einsum('jk,ikj->ij',
                                           x_instances,
                                           np.einsum('ijk,lk->ijl',
                                                     a_inv,
                                                     x_instances)))
    return reward_estim + expl



def reg_loss(theta, x_instance, reward, lamb=c.LAMB):
    '''Standard regularized loss.'''

    loss = np.abs(np.dot(theta, x_instance) - reward) + lamb * np.dot(theta, theta)
    return loss


def entropy_loss(r_est, r_exp, beta):
    '''Entropy loss function used for alpha updates.'''

    return np.exp(-beta * (r_est - r_exp)**2)


def sigmoid_alpha_update(alpha_1, alpha_2, gamma_s, gamma, beta=c.BETA):
    '''Soft version of the hard alpha update rule.
       By adding a KL divergence term to the equation, the solution becomes a sigmoid function.'''

    alpha_1 = alpha_1 * np.exp(-beta * gamma_s)/((np.einsum('mi,mi->i',
                                                            alpha_1,
                                                            np.exp(-beta * gamma_s)
                                                            )[:, np.newaxis] +
                                                  alpha_2 * np.exp(-beta * gamma))[:, 0])
    alpha_2 = 1 - np.clip(np.sum(alpha_1, axis=0), 0, 1)

    if any(a < 0 for a in alpha_2):
        print('negative weight!')

    return alpha_1, alpha_2[:, np.newaxis]


def update_alphas(alpha_1, alpha_2, real_reward, r_loss_1, r_loss_2, beta):
    '''Update alphas, based on the achieved rewards of the bandits.'''

    alpha_1_resc = alpha_1 * entropy_loss(real_reward, r_loss_1, beta)
    alpha_2_resc = alpha_2 * entropy_loss(real_reward, r_loss_2, beta)[:, np.newaxis]
    alpha_1 = alpha_1_resc / (np.sum(alpha_1_resc, axis=0) + alpha_2_resc[:, 0])
    alpha_2 = 1 - np.clip(np.sum(alpha_1, axis=0), 0, 1)
    if any(a < 0 for a in alpha_2):
        print('negative weight!')
    return alpha_1, alpha_2[:, np.newaxis]


def hard_update_alphas(gamma, gamma_s, no_source=c.NO_SOURCES):
    'Hard alpha update rule.'

    ind_min = np.argmin(np.append(gamma_s, gamma.T, axis=0), axis=0)

    alpha = np.zeros((no_source + 1, len(gamma)))

    alpha[ind_min, np.arange(len(gamma))] += 1

    return alpha[:no_source], alpha[no_source:].T


def weighted_training(target,
                      source,
                      theta_estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                      alpha=c.ALPHA,
                      beta=c.BETA,
                      no_sources=c.NO_SOURCES,
                      repeats=c.REPEATS,
                      theta_conf=None,
                      update_rule='hard',
                      probalistic='False',
                      arms_pulled_plot=False):
    '''Training algorithm based on weighted UCB function for linear bandits.'''

    theta_estim /= np.sqrt(np.dot(theta_estim, theta_estim))
    theta_estim = np.tile(theta_estim, (repeats, 1))
    alpha_1 = alpha * np.ones((no_sources, repeats))
    alpha_2 = np.ones((repeats, 1)) - np.sum(alpha_1, axis=0)[:, np.newaxis]
    alpha_evol = np.zeros((repeats, c.EPOCHS))
    target_data = target.target_context
    theta_target = np.tile(target.theta_opt, (repeats, 1))
    # gamma = np.tile(c.GAMMA, (repeats, 1))
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    gamma_s = c.GAMMA * np.ones((no_sources, repeats))
    no_pulls = np.zeros((repeats, c.CONTEXT_SIZE))
    rewards = np.zeros((repeats, c.EPOCHS))
    y_s = np.zeros((no_sources, repeats, c.EPOCHS))

    try:
        a_matrix = c.LAMB * np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))
        a_inv = 1/c.LAMB * np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))
    except ZeroDivisionError:
        print('Lambda cannot be zero!')
        a_matrix = np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))
        a_inv = np.tile(np.identity(c.DIMENSION), (repeats, 1, 1))

    b_vector = np.tile(np.zeros(c.DIMENSION), (repeats, 1))
    regret_evol = np.zeros((repeats, c.EPOCHS))
    x_history = []

    for i in range(0, c.EPOCHS):
        if probalistic:
            index = np.argmax(probalistic_ucb(target_data,
                                              a_inv,
                                              alpha_1,
                                              source,
                                              theta_estim,
                                              gamma_scalar,
                                              gamma_s), axis=1)

        else:
            index = np.argmax(ucb_weighted(target_data,
                                           a_inv,
                                           alpha_1,
                                           alpha_2,
                                           source,
                                           theta_estim,
                                           gamma_scalar,
                                           gamma_s), axis=1)

        index_opt = np.argmax(np.einsum('ij,kj->ik',theta_target, target_data), axis=1)
        instance = target_data[index]
        x_history.append(instance)
        opt_instance = target_data[index_opt]
        r_estim_1 = np.einsum('mij,ij->mi', source, instance)
        # r_estim_1 += 0.1 * (np.ndarray.flatten(gamma_s) *
        #                     np.sqrt(np.einsum('ij,ij->i',
        #                                       instance,
        #                                       np.einsum('ijk,ik->ij',
        #                                                 a_inv,
        #                                                 instance))))
        r_estim_2 = np.einsum('ij,ij->i', theta_estim, instance)
        # r_estim_2 += 0.1 * (np.ndarray.flatten(gamma_scalar) *
        #                     np.sqrt(np.einsum('ij,ij->i',
        #                                       instance,
        #                                       np.einsum('ijk,ik->ij',
        #                                                 a_inv,
        #                                                 instance))))
        noise = c.EPSILON * np.random.normal(scale=c.SIGMA, size=repeats)
        r_real = np.einsum('ij,ij->i', theta_target, instance) + noise
        alpha_evol[:, i] = np.ndarray.flatten(np.asarray(alpha_2))
        rewards[:, i] = r_real
        y_s[:, :, i] = r_estim_1
        # x_theta = np.asarray(x_history)[np.argmax(np.abs(rewards[:, :i + 1] - y_s[:, :, :i + 1]),
        #                                           axis=2),
        #                                 np.arange(repeats)]

        if theta_conf is None:
            gamma_s = np.max(np.abs(rewards[:, :i + 1] - y_s[:, :, :i + 1])/
                                    np.sqrt(np.einsum('lij,lij->li',
                                                      np.asarray(x_history),
                                                      np.asarray(x_history))).T,
                             axis=2) + np.sqrt(np.einsum('mij,mij->mi',
                                                         y_s - rewards,
                                                         y_s - rewards))

            # gamma_s = np.sqrt(np.einsum('mij,mij->mi',
            #                             y_s - rewards,
            #                             y_s - rewards))
        else:
            gamma_s = np.asarray(theta_conf +
                                 np.sqrt(np.einsum('mij,mij->mi',
                                                   y_s - rewards,
                                                   y_s - rewards)))

        gamma_scalar = np.asarray([np.sqrt(c.LAMB) +
                                   np.sqrt(2 * np.log(np.sqrt(np.linalg.det(a_matrix))/
                                                      (np.sqrt(c.LAMB**c.DIMENSION) * c.DELTA)))]).T

        if update_rule=='softmax':
            alpha_1, alpha_2 = update_alphas(alpha_1, alpha_2, r_real, r_estim_1, r_estim_2, beta)

        elif update_rule=='hard':
            alpha_1, alpha_2 = hard_update_alphas(gamma_scalar, gamma_s, no_source=no_sources)

        elif update_rule=='sigmoid':
            alpha_1, alpha_2 = sigmoid_alpha_update(alpha_1,
                                                    alpha_2,
                                                    gamma_s,
                                                    gamma_scalar,
                                                    beta=beta)

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
        no_pulls[np.arange(repeats), index] += 1
        inst_regret = np.einsum('ij,ij->i', theta_target, opt_instance) - r_real
        regret_evol[:, i] = inst_regret
        regr=inst_regret
        print(f"Instant regret = {regr}")

    mean_regret = np.cumsum(regret_evol, axis=1).sum(axis=0)/repeats
    mean_alpha = alpha_evol.sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - np.cumsum(regret_evol, axis=1))**2, axis=0)/repeats)

    if arms_pulled_plot:
        plt.scatter(np.sum(np.einsum('ijk,ik->ij',
                                     target_data,
                                     theta_target),
                           axis=0)/repeats,
                    np.sum(no_pulls, axis=0)/repeats)
        plt.show()
        plt.close()

    return mean_regret, mean_alpha, regret_dev


def compared_alphas(target_class,
                    source,
                    alphas,
                    theta_estim=np.abs(np.random.uniform(size=c.DIMENSION))):
    '''Train on the same context set with differently initialized alphas.'''

    regrets = []
    alpha_evol = []
    regret_devs = []

    for alpha in alphas:
        avr_regret, avr_alpha, std_regret = weighted_training(target_class,
                                                              source,
                                                              theta_estim=theta_estim,
                                                              no_sources=len(source),
                                                              alpha=alpha,
                                                              update_rule='softmax')
        regrets.append(avr_regret)
        alpha_evol.append(avr_alpha)
        regret_devs.append(std_regret)

    return regrets, alpha_evol, regret_devs


def compared_betas(target_class,
                   source,
                   betas,
                   theta_estim=np.abs(np.random.uniform(size=c.DIMENSION)),
                   update_rule='sigmoid',
                   theta_conf=None,
                   probalistic=False):
    '''Train on the same context set with differently initialized alphas.'''

    regrets = []
    alpha_evol = []
    regret_devs = []

    for beta in betas:
        avr_regret, avr_alpha, std_regret = weighted_training(target_class,
                                                              source,
                                                              theta_estim=theta_estim,
                                                              beta=beta,
                                                              no_sources=len(source),
                                                              alpha=1/(len(source)+1),
                                                              update_rule=update_rule,
                                                              probalistic=probalistic,
                                                              theta_conf=theta_conf)

        regrets.append(avr_regret)
        alpha_evol.append(avr_alpha)
        regret_devs.append(std_regret)

    return regrets, alpha_evol, regret_devs
