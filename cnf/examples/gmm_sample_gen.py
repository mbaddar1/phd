import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from distributions.vector_gmm import gen_vec_gaussian_mixture

if __name__ == '__main__':
    k = 2
    d = 2
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = gen_vec_gaussian_mixture(n_components=k, dim=d)
    samples = x.sample_n(1000)

    if d == 1:
        sns.kdeplot(x=samples[:, 0])
    elif d == 2:
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1])
    plt.savefig(f'kde_gmm_k_{k}_d_{d}.png')