from scipy import special as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import synthetic_data as sd
import config as c
matplotlib.use('TKAgg')


TARGET_CLASS = sd.TargetContext()
THETA_SOURCE = sd.source_bandit(TARGET_CLASS.theta_opt)


def ucb_weighted(x_instances, a_inv, alpha_1, alpha_2, theta_s, theta_t, gamma_t, gamma_s=0):
    '''UCB function based on weighted contributions of both domains.'''
    reward_estim_s = alpha_1 * np.einsum('jk,ik->ij', x_instances, theta_s)
    reward_estim_t = alpha_2 * np.einsum('jk,ik->ij', x_instances, theta_t)
    gamma = alpha_1 * gamma_s + alpha_2 * gamma_t
    expl = 0.1 * gamma * np.sqrt(np.einsum('jk,ikj->ij',
                                           x_instances,
                                           np.einsum('ijk,lk->ijl',
                                                     a_inv,
                                                     x_instances)))
    return reward_estim_s + reward_estim_t + expl


def reg_loss(theta, x_instance, reward, lamb=c.LAMB):
    '''Standard regularized loss.'''
    loss = np.abs(np.dot(theta, x_instance) - reward) + lamb * np.dot(theta, theta)
    return loss


def entropy_loss(r_est, r_exp, beta):
    '''Entropy loss function used for alpha updates.'''
    return np.exp(-beta * np.abs(r_est - r_exp)**2)


def sigmoid_alpha_update(alpha_1, gamma_s, gamma, beta=c.BETA):
    '''Soft version of the hard alpha update rule.
       By adding a KL divergence term to the equation, the solution becomes a sigmoid function.'''
    alpha_1 = sp.expit(sp.logit(alpha_1) - beta * (gamma_s - gamma))
    alpha_2 = 1 - alpha_1
    return alpha_1, alpha_2


def update_alphas(alpha_1, alpha_2, real_reward, r_loss_1, r_loss_2, beta):
    '''Update alphas, based on the achieved rewards of the bandits.'''
    alpha_1_resc = alpha_1 * entropy_loss(real_reward, r_loss_1, beta)[:, np.newaxis]
    alpha_2_resc = alpha_2 * entropy_loss(real_reward, r_loss_2, beta)[:, np.newaxis]
    alpha_1 = alpha_1_resc / (alpha_1_resc + alpha_2_resc)
    alpha_2 = 1 - alpha_1
    return alpha_1, alpha_2


def hard_update_alphas(gamma, gamma_s):
    'Hard alpha update rule.'

    alpha_1 = np.argmax([np.ndarray.flatten(gamma_s), np.ndarray.flatten(gamma)],
                        axis=0)[:, np.newaxis]

    alpha_2 = 1 - alpha_1

    return alpha_1, alpha_2


def weighted_training(target=TARGET_CLASS,
                      alpha=c.ALPHA,
                      beta=c.BETA,
                      repeats=c.REPEATS,
                      theta_conf=None,
                      update_rule='hard'):
    '''Training algorithm based on weighted UCB function for linear bandits.'''

    theta_estim = np.abs(np.random.uniform(size=c.DIMENSION))
    theta_estim /= np.sqrt(np.dot(theta_estim, theta_estim))
    theta_estim = np.tile(theta_estim, (repeats, 1))

    alpha_1 = np.tile(alpha, (repeats, 1))
    alpha_2 = np.ones((repeats, 1)) - alpha_1
    alpha_evol = np.zeros((repeats, c.EPOCHS))
    target_data = target.target_context
    theta_target = np.tile(target.theta_opt, (repeats, 1))
    gamma = np.tile(c.GAMMA, (repeats, 1)) #* np.ones(c.context_size)
    gamma_scalar = np.tile(c.GAMMA, (repeats, 1))
    gamma_s = np.tile(c.GAMMA, (repeats, 1)) #* np.ones(c.context_size)
    no_pulls = np.zeros((repeats, c.CONTEXT_SIZE))
    rewards = np.zeros((repeats, c.EPOCHS))
    y_s = np.zeros((repeats, c.EPOCHS))

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
        index = np.argmax(ucb_weighted(target_data,
                                       a_inv,
                                       alpha_1,
                                       alpha_2,
                                       THETA_SOURCE,
                                       theta_estim,
                                       gamma_scalar,
                                       gamma_s), axis=1)
        index_opt = np.argmax(np.einsum('ij,kj->ik',theta_target, target_data), axis=1)
        instance = target_data[index]
        x_history.append(instance)
        opt_instance = target_data[index_opt]
        r_estim_1 = np.einsum('ij,ij->i', THETA_SOURCE, instance) #+ np.ndarray.flatten(gamma_s) * np.sqrt(np.einsum('ij,ij->i', instance, np.einsum('ijk,ik->ij', a_inv, instance)))
        r_estim_2 = np.einsum('ij,ij->i', theta_estim, instance) #+ np.ndarray.flatten(gamma_scalar) * np.sqrt(np.einsum('ij,ij->i', instance, np.einsum('ijk,ik->ij', a_inv, instance)))
        noise = c.EPSILON * np.random.normal(size=repeats)
        r_real = np.einsum('ij,ij->i', theta_target, instance) + noise
        alpha_evol[:, i] = np.ndarray.flatten(np.asarray(alpha_1))
        rewards[:, i] = r_real
        y_s[:, i] = r_estim_1
        x_theta = np.asarray(x_history)[np.argmax(np.abs(rewards[:, :i + 1] - y_s[:, :i + 1]),
                                                  axis=1),
                                        np.arange(repeats)]

        if theta_conf is None:
            gamma_s = np.asarray([np.max(np.abs(rewards[:, :i + 1] - y_s[:, :i + 1]), axis=1)/
                                  np.sqrt(np.einsum('ij,ij->i', x_theta, x_theta))] +
                                 np.sqrt(np.einsum('ij,ij->i',
                                                   y_s - rewards,
                                                   y_s - rewards))).T #* np.ones(c.context_size)
        else:
            gamma_s = np.asarray(theta_conf +
                                 np.sqrt(np.einsum('ij,ij->i',
                                                   y_s - rewards,
                                                   y_s - rewards)))[:,np.newaxis]

        gamma_scalar = np.asarray([np.sqrt(c.LAMB) +
                                   np.sqrt(2 * np.log(np.sqrt(np.linalg.det(a_matrix))/
                                                      (np.sqrt(c.LAMB) * c.DELTA)))]).T
        # gamma = gamma_scalar * np.ones(c.context_size)

        if update_rule=='softmax':
            alpha_1, alpha_2 = update_alphas(alpha_1, alpha_2, r_real, r_estim_1, r_estim_2, beta)

        elif update_rule=='hard':
            alpha_1, alpha_2 = hard_update_alphas(gamma_scalar, gamma_s)

        elif update_rule=='sigmoid':
            alpha_1, alpha_2 = sigmoid_alpha_update(alpha_1, gamma_s, gamma, beta=beta)

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
        # gamma[index] = gamma[index] * 1 / np.sqrt(no_pulls[index])
        inst_regret = np.einsum('ij,ij->i', theta_target, opt_instance) + noise - r_real
        regret_evol[:, i] = inst_regret
        # print("ALPHA SOURCE = {alpha1}\n".format(alpha1=alpha_1))
        regr=inst_regret
        print(f"Instant regret = {regr}")

    mean_regret = regret_evol.sum(axis=0)/repeats
    mean_alpha = alpha_evol.sum(axis=0)/repeats
    regret_dev = np.sqrt(np.sum((mean_regret - regret_evol)**2, axis=0)/repeats)
    # plt.scatter(np.sum(np.einsum('ijk,ik->ij',
    #                              target_data,
    #                              theta_target),
    #                    axis=0)/repeats,
    #             np.sum(no_pulls, axis=0)/repeats)
    # plt.show()
    # plt.close()

    return mean_regret, mean_alpha, regret_dev


def compared_alphas(alphas):
    '''Train on the same context set with differently initialized alphas.'''

    regrets = []
    alpha_evol = []
    regret_devs = []

    for alpha in alphas:

        avr_regret, avr_alpha, std_regret = weighted_training(target=TARGET_CLASS,
                                                              alpha=alpha,
                                                              update_rule='softmax')

        regrets.append(avr_regret)
        alpha_evol.append(avr_alpha)
        regret_devs.append(std_regret)

    return regrets, alpha_evol, regret_devs


def compared_betas(betas, update_rule='sigmoid'):
    '''Train on the same context set with differently initialized alphas.'''

    regrets = []
    alpha_evol = []
    regret_devs = []

    for beta in betas:

        avr_regret, avr_alpha, std_regret = weighted_training(target=TARGET_CLASS,
                                                              beta=beta,
                                                              update_rule=update_rule)

        regrets.append(avr_regret)
        alpha_evol.append(avr_alpha)
        regret_devs.append(std_regret)

    return regrets, alpha_evol, regret_devs
