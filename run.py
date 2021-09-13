import numpy as np
import training as t
import evaluation as e

def main():
    # t.weighted_training()
    regrets, alphas, alpha_evol = t.compared_alphas()
    e.multiple_regret_plots(regrets, alphas)
    e.alpha_plots(alpha_evol)


if __name__ == '__main__':
    main()