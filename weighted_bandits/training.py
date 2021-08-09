import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import synthetic_data as sd
import config as c
import evaluation as e

target = sd.TargetContext()
source = sd.SourceContext()


def ucb_weighted(x, A, alpha_1, alpha_2, theta_s, theta_t, gamma):
    '''UCB function based on weighted contributions of both domains.'''
    reward_estim_s = alpha_1 * np.dot(x, theta_s.T)
    reward_estim_t = alpha_2 * np.dot(x, theta_t.T)
    expl = gamma * np.sqrt(np.diag(np.matmul(x, np.matmul(np.linalg.inv(A), x.T))))
    return reward_estim_s[0] + reward_estim_t[0] + expl


def reg_loss(theta, x, r, lamb):
    '''Standard regularized loss.'''
    loss = np.abs(np.dot(theta, x) - r) + lamb * np.dot(theta, theta)
    return loss


def entropy_loss(r, r_exp):
    '''Entropy loss function used for alpha updates.'''
    return np.exp(-c.beta * np.abs(r - r_exp)**2)


def update_alphas(alpha_1, alpha_2, r, r_loss_1, r_loss_2):
    '''Update alphas, based on the achieved rewards of the bandits.'''
    alpha_1_resc = alpha_1 * entropy_loss(r, r_loss_1)
    alpha_2_resc = alpha_2 * entropy_loss(r, r_loss_2)
    alpha_1 = alpha_1_resc / (alpha_1_resc + alpha_2_resc)
    alpha_2 = alpha_2_resc / (alpha_1_resc + alpha_2_resc)
    return alpha_1, alpha_2


def weighted_training(target=target, source=source):
    '''Training algorithm based on weighted UCB function for linear bandits.'''

    theta_estim = np.random.multivariate_normal(mean=np.zeros(c.dimension),
                                                cov=np.diag(np.ones(c.dimension)),
                                                size=1)[0]
    alpha_1 = 0
    alpha_2 = 1 - alpha_1
    theta_source = source.theta_opt
    target_data = target.target_context
    theta_target = target.theta_opt
    gamma = c.gamma * np.ones(c.context_size)
    no_pulls = np.zeros(c.context_size)

    if c.lamb == 0:
        A_matrix = np.zeros((c.dimension, c.dimension))
    else:
        A_matrix = c.lamb * np.identity(c.dimension)
        
    b_vector = np.zeros(c.dimension)
    regret = 0
    regret_evol = np.zeros(c.epochs)

    for i in range(0, c.epochs):
        index = np.argmax(ucb_weighted(target_data, A_matrix, alpha_1, alpha_2, theta_source, theta_estim, gamma))
        index_opt = np.argmax(np.dot(theta_target, target_data.T))
        instance = target_data[index]
        opt_instance = target_data[index_opt]
        r_estim_1 = np.dot(theta_source, instance)[0]
        r_estim_2 = np.dot(theta_estim, instance)
        r_real = np.dot(theta_target, instance)[0]

        print("TOTAL REWARD LOSS = {loss}\n".format(loss=np.abs(r_real-alpha_1*r_estim_1-alpha_2*r_estim_2)))

        alpha_1, alpha_2 = update_alphas(alpha_1, alpha_2, r_real, r_estim_1, r_estim_2)
        A_matrix += np.outer(instance.T, instance)
        b_vector += r_real * instance
        theta_estim = np.dot(np.linalg.inv(A_matrix), b_vector)
        no_pulls[index] +=1
        gamma[index] = c.gamma * 1 / np.sqrt(no_pulls[index])
        inst_regret = np.dot(theta_target, opt_instance) - np.dot(theta_target, instance)
        regret += inst_regret
        regret_evol[i] = regret
        
        print("ALPHA SOURCE = {alpha1} \nALPHA TARGET = {alpha2} \n".format(alpha1=alpha_1, alpha2=alpha_2))

        print("INSTANT REGRET = {regret}\n".format(regret=inst_regret))

        print("IMMEDIATE REWARD = {reward}\n".format(reward=r_real))
    
    plt.scatter(np.dot(target_data, theta_target.T), no_pulls)
        
    # plt.plot(no_pulls)
    plt.show()




