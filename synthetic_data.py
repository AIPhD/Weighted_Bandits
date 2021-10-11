import random
import numpy as np
import config as c


class TargetContext:
    '''Initializes data set and optimal bandit parameter, implicitly determining rewards as well.'''

    def __init__(self, dimension=c.DIMENSION, context_size=c.CONTEXT_SIZE):

        self.dimension = dimension
        self.context_size = context_size
        self.sigma_t = np.random.uniform(size=self.dimension)
        self.target_context = np.random.multivariate_normal(mean=np.zeros(self.dimension),
                                                            cov=np.diag(self.sigma_t),
                                                            size=self.context_size)
        self.theta_opt = np.abs(np.random.uniform(size=self.dimension))

        if np.dot(self.theta_opt, self.theta_opt) > 1:
            self.theta_opt /= np.sqrt(np.dot(self.theta_opt, self.theta_opt))

        # max_r = np.max(np.dot(self.theta_opt, self.target_context.T))
        # self.theta_opt /= max_r


    def source_bandit(self,
                      dim_align=c.DIMENSION_ALIGN,
                      kappa=c.KAPPA,
                      repeats=c.REPEATS):
        '''Constructing source bandit and norms it properly depending on aligning dimension.'''

        if dim_align == 0:
            theta_s = -self.theta_opt.copy()

        else:
            theta_s = self.theta_opt.copy()
            ind_list = np.sort(random.sample(range(0, self.dimension), dim_align))
            theta_s[ind_list] += (kappa * np.random.multivariate_normal(mean=np.zeros(dim_align),
                                                                        cov=np.identity(dim_align),
                                                                        size=dim_align)[0])
            theta_s /= np.sqrt(np.dot(theta_s, theta_s))
            old_ind = np.delete(np.arange(self.dimension), ind_list)

            if dim_align != self.dimension:
                theta_s[old_ind] = np.abs(np.random.uniform(size=self.dimension - dim_align))
                s_norm = np.sqrt(np.dot(theta_s,
                                        theta_s)/(np.dot(self.theta_opt,
                                                                   self.theta_opt)))

                # if np.dot(theta_s, theta_s) > 1:
                theta_s /= s_norm

        theta_source = np.tile(theta_s, (repeats, 1))

        return theta_source


    def source_bandits(self, no_sources, dim_align=c.DIMENSION_ALIGN):
        '''Construct multiple source bandits with different aligning features.'''

        sources = []
        i = 0

        while i < no_sources:
            sources.append(self.source_bandit(dim_align=dim_align))
            i += 1

        return np.asarray(sources)
