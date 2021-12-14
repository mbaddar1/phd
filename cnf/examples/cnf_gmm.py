import logging

import matplotlib.pyplot as plt
import seaborn
import numpy as np
import torch
from torch import optim
from torchdiffeq import odeint
from tqdm import tqdm
from cnf.examples.cnf_circles import CNF
from utils.dist_gen import gen_torch_gaussian_mixture_1d


def cnf_fit(base_dist: torch.distributions.Distribution,
            target_dist: torch.distributions.Distribution, t0: float, t1: float, in_out_dim: int,
            hidden_dim: int, width: int, lr: float, niters: int, n_samples: int):
    logger_ = logging.getLogger('cnf_fit')
    logger_.info('Starting CNF fit')
    """ Params"""

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    func = CNF(in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width)  # .to(device)
    optimizer = optim.Adam(func.parameters(), lr=lr)
    loss_arr = np.zeros(niters)
    rolling_average_window = 10
    eps = 1e-4
    # train loop
    for itr_idx in tqdm(range(niters)):
        optimizer.zero_grad()
        x = target_dist.sample((n_samples, 1))
        assert x.shape[1] == in_out_dim  # FIXME by design should get dim from data
        log_p_z_init_t1 = get_batched_init_log_p_z(num_samples=n_samples)
        s0 = x, log_p_z_init_t1
        # print(f's0 = {s0}')
        z_t, logp_soln = odeint(
            func,
            s0,
            torch.tensor([t1, t0]).type(torch.FloatTensor),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_soln[-1]
        # dummy1 = base_dist.log_prob(z_t0)
        # dummy2 = logp_diff_t0
        # dummy3 = logp_diff_t0.view(-1)
        logp_x = base_dist.log_prob(z_t0) - logp_diff_t0  # .view(-1)
        loss = -logp_x.mean(0)
        # print('before loss backward')
        loss.backward()
        optimizer.step()
        loss_scalar = loss.detach().numpy()[0]
        loss_arr[itr_idx] = loss_scalar
        if itr_idx % 10 == 0:
            print(f'i = {itr_idx}  loss = {loss_scalar}')
        if itr_idx > rolling_average_window + 1:
            rolling_loss_avg_i = np.nanmean(loss_arr[itr_idx - rolling_average_window:itr_idx])
            rolling_loss_avg_i_minus_1 = np.nanmean(loss_arr[itr_idx - rolling_average_window - 1:itr_idx - 1])
            loss_rolling_avg_abs_diff = np.abs(rolling_loss_avg_i - rolling_loss_avg_i_minus_1)
            if loss_rolling_avg_abs_diff < eps:
                print(f'rolling average loss difference is {loss_rolling_avg_abs_diff}')
                break

    # generate samples
    n_gen = 10000
    z0 = base_dist.sample((n_gen, 1))
    """
    dummy , just to keep things running without touching the core
    """
    log_p_z_init_t0 = get_batched_init_log_p_z(num_samples=n_gen)
    x_gen, _ = odeint(func=func, y0=(z0, log_p_z_init_t0), t=torch.tensor([t0, t1]).type(torch.FloatTensor))
    x_gen = x_gen[-1]
    x_gen_np_arr = x_gen.detach().cpu().numpy().reshape(-1)
    seaborn.kdeplot(x_gen_np_arr)
    plt.savefig('dist.png')


def get_batched_samples_gmm_1d(num_samples):
    gmm = gen_torch_gaussian_mixture_1d()
    samples_ = gmm.sample((num_samples, 1))
    return samples_


# get bach of initial values for log p_z
def get_batched_init_log_p_z(num_samples):
    return torch.zeros(size=(num_samples, 1)).type(torch.FloatTensor)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # params
    t0 = 0
    t1 = 10
    hidden_dim = 32
    width = 64
    in_out_dim = 1
    lr = 1e-3
    n_iters = 100
    n_samples = 512
    # start
    target_dist = gen_torch_gaussian_mixture_1d()
    base_dist = torch.distributions.Normal(loc=0, scale=1)
    cnf_fit(base_dist=base_dist, target_dist=target_dist, t0=t0, t1=t1, in_out_dim=in_out_dim, hidden_dim=hidden_dim,
            width=width,
            lr=1e-3, niters=n_iters, n_samples=n_samples)
