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
from cnf.training import plot_distribution, cnf_fit, generate_samples_cnf
from distributions.vector_gmm import gen_vec_gaussian_mixture
from torchdiffeq import odeint

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
    D_max = 2 if dry_run else 4
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

            x_gen, _ = generate_samples_cnf(cnf_func_fit=cnf_func_fit, base_dist=base_dist, n_samples=batch_size,
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
    logger.info('Job finished successfully')
    # print(results)
