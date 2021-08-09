import numpy as np
import training as t
import evaluation as e

def main():
    # t.weighted_training()
    regrets, alphas = t.compared_alphas()
    # e.multiple_regret_plots(regrets, alphas)


if __name__ == '__main__':
    main()