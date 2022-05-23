import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from torch import optim
from tqdm import tqdm

from cnf.examples.cnf_circles import CNF
from torchdiffeq import odeint
import seaborn as sns


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


# get bach of initial values for logs p_z
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
