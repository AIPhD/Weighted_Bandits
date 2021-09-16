import numpy as np
import training as t
import evaluation as e
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

def main():
    
    betas = [0.1, 10]
    regret_linucb, alpha_none = t.weighted_training(alpha=0, update_rule='softmax')
    regrets_hard, alphas_hard = t.weighted_training()
    regrets, alpha_evol = t.compared_betas(betas, update_rule='softmax')
    regrets_sigmoid, alpha_evol_sigmoid = t.compared_betas(betas, update_rule='sigmoid')
    e.multiple_beta_regret_plots([regret_linucb], plot_label='linUCB')
    e.multiple_beta_regret_plots([regrets_hard], plot_label=r'Hard $\alpha$ update rule')
    e.multiple_beta_regret_plots(regrets, betas, plot_label='softmax')
    e.multiple_beta_regret_plots(regrets_sigmoid, betas, plot_label='sigmoid')

    plt.show()
    plt.savefig('/home/steven/weighted_bandits/plots/regret_comparison.png')
    plt.close()
    e.alpha_plots([alphas_hard])
    e.alpha_plots(alpha_evol, betas)


if __name__ == '__main__':
    main()