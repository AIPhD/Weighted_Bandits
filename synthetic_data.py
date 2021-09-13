import config as c
import numpy as np


class TargetContext:

    def __init__(self):
        '''Initializes data set and optimal bandit parameter, this way rewards are implicitly determined as well.'''

        self.sigma_T=np.random.uniform(size=c.dimension)
        self.target_context = np.random.multivariate_normal(mean=np.zeros(c.dimension), 
                                                            cov=np.diag(self.sigma_T),
                                                            size=c.context_size)
        self.theta_opt = np.abs(np.random.uniform(size=c.dimension))
        self.theta_opt /= np.sqrt(np.dot(self.theta_opt, self.theta_opt))

        # max_r = np.max(np.dot(self.theta_opt, self.target_context.T))
        # self.theta_opt /= max_r


                                                