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


def gen_vec_gaussian_mixture_k3_d2(n_components, dim):
    scale_tril_list = []
    for i in range(n_components):
        t = torch.randn(dim, dim)
        t = torch.tril(t)
        t.fill_diagonal_(np.random.uniform(0.1, 10))
        scale_tril_list.append(t)
    scale_tril_tensor = torch.stack(scale_tril_list)
    comp = torch.distributions.Independent(
        torch.distributions.MultivariateNormal(loc=20 + torch.randn(n_components, dim) * 20,
                                               scale_tril=scale_tril_tensor), 0)

    mix = torch.distributions.Categorical(torch.ones(n_components))
    gmm_ = torch.distributions.MixtureSameFamily(mix, comp)
    return gmm_


def gen_vec_gaussian_mixture(n_components, dim):
    scale_tril_list = []
    loc_list = []
    loc0 = torch.tensor([1.0] * dim)
    for i in range(n_components):
        t = torch.randn(dim, dim)
        t = torch.tril(t)
        t.fill_diagonal_(np.random.uniform(0.1, 10))
        scale_tril_list.append(t)
        loc_ = loc0 + (i + 1) * 10
        loc_list.append(loc_)

    scale_tril_tensor = torch.stack(scale_tril_list)
    loc_tensor = torch.stack(loc_list)
    comp = torch.distributions.Independent(
        torch.distributions.MultivariateNormal(loc=loc_tensor, scale_tril=scale_tril_tensor), 0)

    mix = torch.distributions.Categorical(torch.ones(n_components))
    gmm_ = torch.distributions.MixtureSameFamily(mix, comp)
    return gmm_


# def gen_torch_gaussian_mixture_1d():
#     mix = torch.distributions.Categorical(torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))
#     comp = torch.distributions.Normal(torch.tensor([-10.0, 0, 10.0]), torch.tensor([2.0, 3.0, 2.0]))
#     gmm_ = torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
#     return gmm_


# need revisit
# def gen_gmm_1d(n_components: int, dim: int):
#     """
#     :param n_components:
#     :param dim:
#     :return:
#     """
#     mix = torch.distributions.Categorical(torch.ones(n_components, ))
#     comp = torch.distributions.Independent(
#         torch.distributions.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim)), 1)
#     dummy1 = torch.distributions.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim))
#
#     gmm_ = torch.distributions.MixtureSameFamily(mix, comp)
#     x = gmm_.sample_n(100)  # dummy line , for illustration only
#     """
#     x is of shape (N,D) where N is number of samples and D is the dimension for each 1xD generated vector
#     """
#     return gmm_


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Main')
    # need revisit
    for i in range(10):
        n_sample = 10000
        N_components = 6
        Nd = 1
        gmm = gen_vec_gaussian_mixture(n_components=N_components, dim=1)
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
