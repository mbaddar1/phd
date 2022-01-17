import datetime
import logging
import time

import numpy as np
import torch
from torch import optim
from torchdiffeq import odeint
from tqdm import tqdm

from cnf.examples.cnf_circles import CNF
from distributions import gen_vec_gaussian_mixture

# TODO
"""
Expressive power measuring 
1. Train a CNF to model GMM for X random variable where X in R^(1xD) 
2. Generate N samples, each with dimension d from trained NF (based on random latents)
3. Generate N* samples from the original GMM model 
4. Measure (2) : Calculate log likelihood for N given CNF 
5. Observe the relation between measure (1) and (2) 
6. Repeat the Experiment for Matrix / Tensor Valued Random Variables 
i.e. X in R^(D_1xD_2) or R^(D_1xD_2x...xD_nd) 
# https://arxiv.org/pdf/1911.02915.pdf 
"""

"""
Got error
Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
Possible cause : extensive memory usage
https://stackoverflow.com/questions/43268156/process-finished-with-exit-code-137-in-pycharm

Increase Ubuntu 20.04 swap
https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04 

cmd

size="32G" && file_swap=/swapfile_$size.img && sudo touch $file_swap && 
sudo fallocate -l $size /$file_swap && sudo mkswap /$file_swap && sudo swapon -p 20 /$file_swap 

sudo chmod 0600 /swapfile_32G.img 

check swap size
cat /proc/swaps

"""


def cnf_fit(base_dist: torch.distributions.Distribution,
            target_dist: torch.distributions.Distribution, t0: float, t1: float, in_out_dim: int,
            hidden_dim: int, width: int, lr: float, train_batch_size: int, niters: int):
    logger_ = logging.getLogger('cnf_fit')
    logger_.info('Starting CNF fit')
    """ Params"""

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    cnf_func_instance = CNF(in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width)  # .to(device)
    optimizer = optim.Adam(cnf_func_instance.parameters(), lr=lr)
    loss_arr = np.zeros(niters)
    rolling_average_window = 3
    eps = 1e-10
    # train loop
    final_loss = None
    for itr_idx in tqdm(range(niters)):
        optimizer.zero_grad()
        x = target_dist.sample_n(train_batch_size)
        assert x.shape[1] == in_out_dim  # FIXME by design should get dim from data
        log_p_z_init_t1 = get_batched_init_log_p_z(num_samples=train_batch_size)
        s0 = x, log_p_z_init_t1
        # print(f's0 = {s0}')
        z_t, logp_soln = odeint(
            cnf_func_instance,
            s0,
            torch.tensor([t1, t0]).type(torch.FloatTensor),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        z_t0, logp_diff_t0 = z_t[-1], logp_soln[-1]
        dummy1 = base_dist.log_prob(z_t0)
        dummy2 = logp_diff_t0
        dummy3 = logp_diff_t0.view(-1)
        logp_x = base_dist.log_prob(z_t0) - logp_diff_t0.view(-1)  # .view(-1)
        loss = -logp_x.mean(0)
        loss.backward()
        optimizer.step()
        loss_scalar = loss.detach().numpy()
        loss_arr[itr_idx] = loss_scalar
        final_loss = loss_scalar
        if itr_idx % 10 == 0:
            logger_.info(f'\ni = {itr_idx}  loss = {loss_scalar}\n')
        if itr_idx > rolling_average_window + 1:
            rolling_loss_avg_i = np.nanmean(loss_arr[itr_idx - rolling_average_window:itr_idx])
            rolling_loss_avg_i_minus_1 = np.nanmean(loss_arr[itr_idx - rolling_average_window - 1:itr_idx - 1])
            loss_rolling_avg_abs_diff = np.abs(rolling_loss_avg_i - rolling_loss_avg_i_minus_1)
            if loss_rolling_avg_abs_diff < eps:
                logger_.info(f'rolling average loss difference is {loss_rolling_avg_abs_diff}')
                break
    logger_.info(f'final loss at iter = {niters} = {final_loss}')
    return cnf_func_instance, final_loss


def generate_samples_cnf(cnf_func, base_dist, n_samples):
    z0 = base_dist.sample_n(n_samples)
    assert isinstance(cnf_func, CNF)
    log_p_z_init_t0 = get_batched_init_log_p_z(num_samples=n_samples)
    x_gen, _ = odeint(func=cnf_func, y0=(z0, log_p_z_init_t0), t=torch.tensor([t0, t1]).type(torch.FloatTensor))
    x_gen = x_gen[-1]
    return x_gen


# get bach of initial values for log p_z
def get_batched_init_log_p_z(num_samples):
    return torch.zeros(size=(num_samples, 1)).type(torch.FloatTensor)


def plot_distribution(x: torch.Tensor):
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # params
    t0 = 0
    t1 = 10
    hidden_dim = 32
    width = 64
    train_batch_size = 512
    X_dim_max = 20
    lr = 1e-3
    n_iters = 100
    n_components_gmm = 3

    # experiment with different dimensions of data
    per_dim_count = 3
    file = open(f"experiment_{datetime.datetime.now().isoformat()}.log", "w")
    file.write("dim,avg_train_time_sec,out_of_sample_loss,in_sample_loss\n")
    file.flush()
    for X_dim in np.arange(1, X_dim_max + 1):
        logger.info(f'X_dim = {X_dim} out of {X_dim_max}')
        target_dist = gen_vec_gaussian_mixture(n_components=n_components_gmm, dim=X_dim)
        base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(X_dim),
                                                           scale_tril=torch.diag(torch.ones(X_dim)))
        time_diff_sum = 0
        log_prob_sum = 0
        in_out_sample_loss_diff_sum = 0
        in_sample_loss_sum = 0
        # for each dim , repeat a couple of times to take the average per each dim
        for j in range(per_dim_count):
            logger.info(f'for X_dim = {X_dim}, iter {j + 1} out of {per_dim_count}')
            start_time = time.time()
            cnf_func_fit, final_loss = cnf_fit(base_dist=base_dist, target_dist=target_dist, t0=t0, t1=t1,
                                               in_out_dim=X_dim,
                                               hidden_dim=hidden_dim, width=width, lr=1e-3, niters=n_iters,
                                               train_batch_size=train_batch_size)
            end_time = time.time()

            time_diff_sec = end_time - start_time
            logger.info(
                f'n_components_gmm = {n_components_gmm}, X_dim = {X_dim}, time_diff_sec = {time_diff_sec}')

            n_test_samples = 1000
            gen_samples = generate_samples_cnf(cnf_func=cnf_func_fit, base_dist=base_dist, n_samples=n_test_samples)
            log_prob_test = target_dist.log_prob(x=torch.tensor(gen_samples))
            log_prob_test_avg = log_prob_test.mean(0).detach().numpy()
            logger.info(f'In-sample loss = {final_loss} , out-of-sample = {-log_prob_test_avg} , difference = '
                        f'{np.abs(log_prob_test_avg + final_loss)}')
            logger.info(
                f'n_components_gmm = {n_components_gmm}, X_dim = {X_dim}, avg_log_prob = '
                f'{log_prob_test.mean(0).detach().numpy()}')
            time_diff_sum += time_diff_sec
            log_prob_sum += log_prob_test.mean(0).detach().numpy()
            # in_out_sample_loss_diff_sum += np.abs(log_prob_test_avg + final_loss)
            in_sample_loss_sum += final_loss
        file.write(
            f"{X_dim},{time_diff_sum / per_dim_count},{-log_prob_sum / per_dim_count},"
            f"{in_sample_loss_sum / per_dim_count}\n")
        file.flush()

    file.flush()
    file.close()
    # print(results)
