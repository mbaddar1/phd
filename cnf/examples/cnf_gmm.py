import logging

import torch
from torch import optim
from torchdiffeq import odeint
from tqdm import tqdm

from cnf.examples.cnf_circles import RunningAverageMeter, CNF
from utils.dist_gen import gen_torch_gaussian_mixture_1d


def cnf_fit(base_dist: torch.distributions.Distribution,
            target_dist: torch.distributions.Distribution, in_out_dim: int,
            hidden_dim: int, width: int, lr: float, niters: int, n_samples: int):
    logger_ = logging.getLogger('cnf_fit')
    logger_.info('Starting CNF fit')
    """ Params"""

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    func = CNF(in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width)  # .to(device)

    optimizer = optim.Adam(func.parameters(), lr=lr)

    loss_meter = RunningAverageMeter()  # TODO what is this ??

    # train loop
    for itr in tqdm(range(niters)):
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
        loss_meter.update(loss.item())
        print(f'iter = {itr} , loss = {loss.detach().numpy()[0]}')


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
    n_iters = 10000
    n_samples = 512
    # start
    target_dist = gen_torch_gaussian_mixture_1d()
    base_dist = torch.distributions.Normal(loc=0, scale=1)
    cnf_fit(base_dist=base_dist, target_dist=target_dist, in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width,
            lr=1e-3, niters=n_iters, n_samples=n_samples)
