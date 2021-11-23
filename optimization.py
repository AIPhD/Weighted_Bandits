import numpy as np
import scipy.optimize as opt


def least_squares_bounded_opt(x_data, y_rewards, theta_s, theta_t):
    '''Optimize least squares problem with linear constraints.'''
    delta_theta = theta_s - theta_t
    repeats = len(theta_t)
    dimension = len(theta_t[0])
    w_1 = np.zeros((repeats, dimension, dimension))
    w_2 = np.zeros((repeats, dimension, dimension))
    for i in range(0, len(x_data[0, :, 0])):
        x_new = np.einsum('jk,kl->jl', x_data[:, i, :], np.diag(delta_theta[i]))
        y_new = np.einsum('jk,k->j', x_data[:, i, :], theta_t[i]) - y_rewards[i]
        w_new = np.diag(opt.lsq_linear(x_new,
                                       y_new,
                                       bounds=(np.zeros(dimension),
                                               np.ones(dimension))).x)
        w_1[i] = w_new
        w_2[i] = np.identity(dimension) - w_new
    return w_1, w_2


def least_squares_opt(x_data, y_rewards, theta_s, theta_t):
    '''Optimize least squares problem with linear constraints.'''
    delta_theta = theta_s - theta_t
    dimension = len(theta_t[0])
    repeats = len(theta_t)
    w_1 = np.zeros((repeats, dimension, dimension))
    w_2 = np.zeros((repeats, dimension, dimension))
    for i in range(0, len(x_data[0, :, 0])):
        x_new = np.einsum('jk,kl->jl', x_data[:, i, :], np.diag(delta_theta[i]))
        y_new = np.einsum('jk,k->j', x_data[:, i, :], theta_t[i]) - y_rewards[i]
        w_new = np.diag(np.einsum('jk,k->j',
                                  np.linalg.inv(np.cov(x_new)),
                                  np.einsum('jk,j->k', x_new, y_new)))
        w_1[i] = w_new
        w_2[i] = np.identity(dimension) - w_new
    return w_1, w_2
