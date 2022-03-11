import datetime
import logging
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import TruncatedSVD
from torch import optim
from tqdm import tqdm

from cnf.examples.cnf_circles import CNF
from distributions.vector_gmm import gen_vec_gaussian_mixture
from torchdiffeq import odeint

# TODO
"""
1- understand the meaning of t0=0, t1=10, how discretization happens ?
"""
"""
Tech. Issues 
Got error
Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
Possible cause : extensive memory usage
https://stackoverflow.com/questions/43268156/process-finished-with-exit-code-137-in-pycharm

Increase Ubuntu 20.04 swap
https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04 

cmd

size="64G" && file_swap=/swapfile_$size.img && sudo touch $file_swap && 
sudo fallocate -l $size /$file_swap && sudo mkswap /$file_swap && sudo swapon -p 20 /$file_swap 

sudo chmod 0600 /swapfile_64G.img 

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
        (z_t, logp_soln), _ = odeint(
            cnf_func_instance,
            s0,
            torch.tensor([t1, t0]).type(torch.FloatTensor),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5', is_f_t_evals=False
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


def generate_samples_cnf(cnf_func, base_dist, n_samples, t0, t1, is_f_t_evals):
    # timestamp = datetime.datetime.now().isoformat()
    z0 = base_dist.sample_n(n_samples)
    assert isinstance(cnf_func, CNF)
    log_p_z_init_t0 = get_batched_init_log_p_z(num_samples=n_samples)

    (x_gen, _), ft_numeric = odeint(func=cnf_func, y0=(z0, log_p_z_init_t0),
                                    t=torch.tensor([t0, t1]).type(torch.FloatTensor), is_f_t_evals=is_f_t_evals)

    ft_dict = ft_numeric.convert_to_dict()
    # pickle_name = f'ft_n_{n_samples}_K_{gmm_k}_D_{list(ft_numeric.shapes[0])[1]}_niter_{n_iters}_timestamp_{timestamp}.pkl'
    # pickle.dump(obj=ft_dict, file=open(os.path.join(model_dir, pickle_name), 'wb'))
    # # ft_dict_loaded = pickle.load(file=open(os.path.join(model_dir, pickle_name), 'rb'))
    x_gen = x_gen[-1]
    return x_gen, ft_dict


# get bach of initial values for log p_z
def get_batched_init_log_p_z(num_samples):
    return torch.zeros(size=(num_samples, 1)).type(torch.FloatTensor)


def plot_distribution(x: torch.Tensor, filename: str):
    plt.clf()
    d = list(x.size())[1]
    if d == 1:
        sns.kdeplot(x=x[:, 0].detach().numpy())
    elif d == 2:
        sns.kdeplot(x=x[:, 0].detach().numpy(), y=x[:, 1].detach().numpy())
    else:
        svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
        x_svd = svd.fit_transform(x.detach().numpy())
        sns.kdeplot(x=x_svd[:, 0], y=x_svd[:, 1])
    plt.savefig(filename)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    saved_models_path = 'models'
    timestamp = datetime.datetime.now().isoformat()
    dry_run = False
    # params
    t0 = 0
    t1 = 10
    hidden_dim = 64
    width = 1024
    train_batch_size = 500
    D_max = 2 if dry_run else 2
    lr = 1e-3
    n_iters = 10 if dry_run else 3000
    batch_size = 10 if dry_run else 500
    K = 3  # num GMM components
    is_f_t_evals = True
    # experiment with different dimensions of data
    per_dim_count = 1
    plot_dir = 'plots'
    # seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    ###
    file = open(f"./experiments_logs/experiment_{datetime.datetime.now().isoformat()}.log", "w")
    file.write("dim,avg_train_time_sec,out_of_sample_loss,in_sample_loss\n")
    file.flush()
    for D in np.arange(D_max, D_max + 1):

        logger.info(f'X_dim = {D} out of {D_max}')
        target_dist = gen_vec_gaussian_mixture(n_components=K, dim=D)
        sample_true_input = target_dist.sample_n(batch_size)
        plot_distribution(sample_true_input,
                          os.path.join(plot_dir, f'input_kde_K_{K}_D_{D}_n_{batch_size}_{timestamp}.png'))
        base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D),
                                                           scale_tril=torch.diag(torch.ones(D)))
        time_diff_sum = 0
        log_prob_sum = 0
        in_out_sample_loss_diff_sum = 0
        in_sample_loss_sum = 0
        # for each dim , repeat a couple of times to take the average per each dim
        for j in range(per_dim_count):
            logger.info(f'for X_dim = {D}, iter {j + 1} out of {per_dim_count}')
            start_time = time.time()
            cnf_func_fit, final_loss = cnf_fit(base_dist=base_dist, target_dist=target_dist, t0=t0, t1=t1,
                                               in_out_dim=D,
                                               hidden_dim=hidden_dim, width=width, lr=1e-3, niters=n_iters,
                                               train_batch_size=train_batch_size)
            model_name = f'cnf_func_fit_gmm_K_{K}_D_{D}_niters_{n_iters}_{timestamp}.pkl'
            torch.save(cnf_func_fit, os.path.join(saved_models_path, model_name))
            end_time = time.time()

            time_diff_sec = end_time - start_time
            logger.info(
                f'n_components_gmm = {K}, X_dim = {D}, time_diff_sec = {time_diff_sec}')

            x_gen,_ = generate_samples_cnf(cnf_func=cnf_func_fit, base_dist=base_dist, n_samples=batch_size,
                                         t0=t0, t1=t1, is_f_t_evals=is_f_t_evals)
            plot_distribution(x_gen,
                              os.path.join(plot_dir, f'output_kde_K_{K}_D_{D}_niters_{n_iters}_{timestamp}.png'))
            log_prob_test = target_dist.log_prob(x=torch.tensor(x_gen))
            log_prob_test_avg = log_prob_test.mean(0).detach().numpy()
            logger.info(f'In-sample loss = {final_loss} , out-of-sample = {-log_prob_test_avg} , difference = '
                        f'{np.abs(log_prob_test_avg + final_loss)}')
            logger.info(
                f'n_components_gmm = {K}, X_dim = {D}, avg_log_prob = '
                f'{log_prob_test.mean(0).detach().numpy()}')
            time_diff_sum += time_diff_sec
            log_prob_sum += log_prob_test.mean(0).detach().numpy()
            in_sample_loss_sum += final_loss

            ## Generate samples ##
            # FIXME sometimes I think this step take a lot of memory and crashes the process, so it is safer
            # to separate training from sample generation
            # cnf_func_fit.log_f_t = True
            # generate_samples_cnf(cnf_func=cnf_func_fit, base_dist=base_dist, n_samples=batch_size, t0=t0, t1=t1,
            #                      is_f_t_evals=is_f_t_evals, gmm_k=K, n_iters=n_iters, model_dir=saved_models_path)
            # use the cnf_func pkl file

        file.write(
            f"{D},{time_diff_sum / per_dim_count},{-log_prob_sum / per_dim_count},"
            f"{in_sample_loss_sum / per_dim_count}\n")
        file.flush()

    file.flush()
    file.close()
    # print(results)
