import config as c
import numpy as np



class SourceContext:

    def __init__(self):

        self.sigma_S=np.random.uniform(size=c.dimension)
        self.source_context = np.random.multivariate_normal(mean=np.zeros(c.dimension), 
                                                            cov=np.diag(self.sigma_S),
                                                            size=c.context_size)
        self.theta_opt = np.abs(np.random.multivariate_normal(mean=np.zeros(c.dimension),
                                                              cov=np.diag(np.ones(c.dimension)),
                                                              size=1))
        
        max_r = np.max(np.dot(self.theta_opt, self.source_context.T))
        self.theta_opt *= 500
                                    


class TargetContext:

    def __init__(self):

        self.sigma_T=np.random.uniform(size=c.dimension)
        self.target_context = np.random.multivariate_normal(mean=np.zeros(c.dimension), 
                                                            cov=np.diag(self.sigma_T),
                                                            size=c.context_size)
        self.theta_opt = np.abs(np.random.multivariate_normal(mean=np.zeros(c.dimension),
                                                              cov=np.diag(np.ones(c.dimension)),
                                                              size=1))

        max_r = np.max(np.dot(self.theta_opt, self.target_context.T))
        self.theta_opt *= 5


                                                