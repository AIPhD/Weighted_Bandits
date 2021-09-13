import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import synthetic_data as sd
import config as c
import evaluation as e


target = sd.TargetContext()
theta_source = np.tile(target.theta_opt + 
                                    c.kappa * np.random.multivariate_normal(mean=np.zeros(c.dimension), 
                                                                            cov=np.identity(c.dimension), 
                                                                            size=c.dimension)[0], (c.repeats, 1))


def ucb_weighted(x, A_inv, alpha_1, alpha_2, theta_s, theta_t, gamma_T, gamma_S=0):
    '''UCB function based on weighted contributions of both domains.'''
    reward_estim_s = alpha_1 * np.einsum('ijk,ik->ij', x, theta_s)
    reward_estim_t = alpha_2 * np.einsum('ijk,ik->ij', x, theta_t)
    gamma = alpha_1 * gamma_S + alpha_2 * gamma_T
    expl = gamma * np.sqrt(np.einsum('ijk,ikj->ij', x, np.einsum('ijk,ilk->ijl', A_inv, x)))     
    return reward_estim_s + reward_estim_t + expl


def reg_loss(theta, x, r, lamb):
    '''Standard regularized loss.'''
    loss = np.abs(np.dot(theta, x) - r) + lamb * np.dot(theta, theta)
    return loss


def entropy_loss(r, r_exp):
    '''Entropy loss function used for alpha updates.'''
    return np.exp(-c.beta * np.abs(r - r_exp)**2)  # unclear if quadratic or not


def update_alphas(alpha_1, alpha_2, r, r_loss_1, r_loss_2):
    '''Update alphas, based on the achieved rewards of the bandits.'''
    alpha_1_resc = alpha_1 * entropy_loss(r, r_loss_1)[:, np.newaxis]
    alpha_2_resc = alpha_2 * entropy_loss(r, r_loss_2)[:, np.newaxis]
    # if alpha_1_resc + alpha_2_resc == 0:
      #   print("Stop")
    alpha_1 = alpha_1_resc / (alpha_1_resc + alpha_2_resc)
    alpha_2 = 1 - alpha_1
    return alpha_1, alpha_2


def new_update_alphas(y_S, y_T, gamma):

    if np.sqrt(4 + np.dot(y_S - y_T, y_S - y_T)) < gamma:
        alpha_1 = 1
    else:
        alpha_1 = 0

    alpha_2 = 1 - alpha_1

    return alpha_1, alpha_2
    


def weighted_training(target=target, alpha=c.alpha, repeats=c.repeats):
    '''Training algorithm based on weighted UCB function for linear bandits.'''

    theta_estim = np.reshape(np.repeat(np.random.multivariate_normal(mean=np.zeros(c.dimension),
                                                                     cov=np.diag(np.ones(c.dimension)),
                                                                     size=1)[0], c.repeats, axis=0), (repeats, c.dimension))

    alpha_1 = np.tile(alpha, (repeats, 1))
    alpha_2 = np.ones((repeats, 1)) - alpha_1
    alpha_evol = np.zeros((repeats, c.epochs))
    target_data = np.tile(target.target_context, (repeats, 1, 1))
    theta_target = np.tile(target.theta_opt, (repeats, 1))
    
    gamma = np.tile(c.gamma, (repeats, 1)) #* np.ones(c.context_size)
    gamma_scalar = np.tile(c.gamma, (repeats, 1))
    gamma_S = np.tile(c.gamma, (repeats, 1)) #* np.ones(c.context_size)
    no_pulls = np.zeros((repeats, c.context_size))
    rewards = np.zeros((repeats, c.epochs))
    y_S = np.zeros((repeats, c.epochs))

    try:
        A_matrix = c.lamb * np.tile(np.identity(c.dimension), (repeats, 1, 1))
        A_inv = 1/c.lamb * np.tile(np.identity(c.dimension), (repeats, 1, 1))
    except:
        c.lamb = 0
        print('Lambda canot be zero!')
         
    b_vector = np.tile(np.zeros(c.dimension), (repeats, 1))
    regret_evol = np.zeros((repeats, c.epochs))
    x_history = []

    for i in range(0, c.epochs):
        index = np.argmax(ucb_weighted(target_data, 
                                       A_inv, 
                                       alpha_1, 
                                       alpha_2, 
                                       theta_source, 
                                       theta_estim, 
                                       gamma_scalar,
                                       gamma_S), axis=1)
        index_opt = np.argmax(np.einsum('ij,ikj->ik',theta_target, target_data), axis=1)
        instance = target_data[np.arange(repeats), index]
        x_history.append(instance)
        opt_instance = target_data[np.arange(repeats), index_opt]
        r_estim_1 = np.einsum('ij,ij->i', theta_source, instance)
        r_estim_2 = np.einsum('ij,ij->i', theta_estim, instance)    # TODO: add exploration to reward
        noise = c.epsilon * np.random.normal(size=repeats)
        r_real = np.einsum('ij,ij->i', theta_target, instance) + noise


        # print("TOTAL REWARD LOSS = {loss}\n".format(loss=np.abs(r_real-alpha_1*r_estim_1-alpha_2*r_estim_2)))

        alpha_evol[:, i] = np.ndarray.flatten(np.asarray(alpha_1))
        rewards[:, i] = r_real
        x_theta = np.asarray(x_history)[np.argmax(np.abs(rewards[:, :i + 1] - y_S[:, :i + 1]), axis=1), 
                                        np.arange(repeats)]
        gamma_S = np.asarray([np.max(np.abs(rewards[:, :i + 1] - y_S[:, :i + 1]), axis=1)/np.sqrt(np.einsum('ij,ij->i', x_theta, x_theta))]).T #* np.ones(c.context_size)
        gamma_scalar = np.asarray([np.sqrt(c.lamb) + np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A_matrix))
                                                                        /(np.sqrt(c.lamb) * c.delta)))]).T * 0.1
        # gamma = gamma_scalar * np.ones(c.context_size)
        y_S[:, i] = r_estim_1
        alpha_1, alpha_2 = update_alphas(alpha_1, alpha_2, r_real, r_estim_1, r_estim_2)
        # alpha_1, alpha_2 = new_update_alphas(y_S, rewards, gamma_scalar)
        A_matrix += np.einsum('ij,ik->ijk', instance, instance)
        
        A_inv -= np.einsum('ijk,ikl->ijl', A_inv, np.einsum('ijk,ikl->ijl', np.einsum('ij,ik->ijk', instance, instance), A_inv))/(1 + np.einsum('ij,ij->i', instance, np.einsum('ijk,ik->ij', A_inv, instance)))[:, np.newaxis, np.newaxis]
        b_vector += r_real[:, np.newaxis] * instance
        theta_estim = np.einsum('ijk,ik->ij', A_inv, b_vector)
        # no_pulls[index] +=1
        # gamma[index] = gamma[index] * 1 / np.sqrt(no_pulls[index])
        inst_regret = np.einsum('ij,ij->i', theta_target, opt_instance) - r_real
        regret_evol[:, i] = inst_regret
        
        print("ALPHA SOURCE = {alpha1} \nALPHA TARGET = {alpha2} \n".format(alpha1=alpha_1, alpha2=alpha_2))

    
    # plt.scatter(np.dot(target_data, theta_target.T), no_pulls)
    # plt.show()
    # plt.close()

    return regret_evol.sum(axis=0)/repeats, alpha_evol.sum(axis=0)/repeats


def compared_alphas():
    '''Train on the same context set with differently initialized alphas.'''

    alphas = [0, 0.5, 0.99]
    regrets = []
    alpha_evol = []

    for alpha in alphas:

        avr_regret, avr_alpha = weighted_training(target=target, alpha=alpha)

        regrets.append(avr_regret)
        alpha_evol.append(avr_alpha)

    return regrets, alphas, alpha_evol

