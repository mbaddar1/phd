"""
Script to generate different synthetic distributions
"""
import logging

import seaborn
import torch
import torch.distributions as D
from matplotlib import pyplot as plt

"""
Gaussian Mixture
"""


def gen_torch_gaussian_mixture_1d():
    mix = D.Categorical(torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))
    comp = D.Normal(torch.tensor([-10.0, 0, 10.0]), torch.tensor([2.0, 3.0, 2.0]))
    gmm = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    return gmm


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Main')
    n_sample = 10000
    gmm = gen_torch_gaussian_mixture_1d()
    samples = gmm.sample_n(n_sample)
    plt.hist(samples.detach().cpu().numpy(), density=True, bins=100)
    seaborn.kdeplot(samples.detach().cpu().numpy())
    plt.savefig('kde.png')
