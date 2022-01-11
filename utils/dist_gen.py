"""
Script to generate different synthetic distributions
"""
import logging

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt

"""
Gaussian Mixture
"""


def gen_torch_gaussian_mixture_1d():
    mix = torch.distributions.Categorical(torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))
    comp = torch.distributions.Normal(torch.tensor([-10.0, 0, 10.0]), torch.tensor([2.0, 3.0, 2.0]))
    gmm_ = torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    return gmm_


def gen_gmm_Nd(n_components: int, dim: int):
    """
    :param n_components:
    :param dim:
    :return:
    """
    mix = torch.distributions.Categorical(torch.ones(n_components, ))
    comp = torch.distributions.Independent(
        torch.distributions.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim)), 1)
    gmm_ = torch.distributions.MixtureSameFamily(mix, comp)
    x = gmm_.sample_n(100)
    return gmm_


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Main')
    for i in range(10):
        n_sample = 10000
        N_components = 3
        Nd = 1
        gmm = gen_gmm_Nd(n_components=N_components, dim=1)
        samples = gmm.sample_n(n_sample)
        samples_ndarray = samples.detach().cpu().numpy()
        if Nd == 1:
            samples_ndarray = np.reshape(a=samples_ndarray, newshape=-1)
            seaborn.kdeplot(samples_ndarray)
            # plt.show()

            plt.savefig(f'kde{i}.png')
            plt.clf()

        else:
            print(f'plot for d > 1 not supported yet')
