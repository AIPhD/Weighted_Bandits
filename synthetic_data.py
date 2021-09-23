import numpy as np
import config as c


class TargetContext:
    '''Initializes data set and optimal bandit parameter, implicitly determining rewards as well.'''

    def __init__(self):

        self.sigma_t=np.random.uniform(size=c.DIMENSION)
        self.target_context = np.random.multivariate_normal(mean=np.zeros(c.DIMENSION),
                                                            cov=np.diag(self.sigma_t),
                                                            size=c.CONTEXT_SIZE)
        self.theta_opt = np.abs(np.random.uniform(size=c.DIMENSION))
        self.theta_opt /= np.sqrt(np.dot(self.theta_opt, self.theta_opt))

        # max_r = np.max(np.dot(self.theta_opt, self.target_context.T))
        # self.theta_opt /= max_r


def source_bandit(theta_opt,
                  dim_align=c.DIMENSION_ALIGN,
                  dimension=c.DIMENSION,
                  kappa=c.KAPPA,
                  repeats=c.REPEATS):
    '''Constructing source bandit and norms it properly depending on aligning dimension.'''

    if dim_align == 0:
        theta_s = -theta_opt

    else:
        theta_s = theta_opt
        theta_s[:dim_align] += (kappa * np.random.multivariate_normal(mean=np.zeros(dim_align),
                                                                      cov=np.identity(dim_align),
                                                                      size=dim_align)[0])

        if dim_align != dimension:
            theta_s[dim_align:] = np.abs(np.random.uniform(size=dimension - dim_align))
            s_norm = np.dot(theta_s[dim_align:],
                            theta_s[dim_align:])/np.sqrt(1 -
                                                            np.dot(theta_s[:dim_align],
                                                                   theta_s[:dim_align]))
            theta_s[dim_align:] /= s_norm

    theta_source = np.tile(theta_s, (repeats, 1))

    return theta_source
