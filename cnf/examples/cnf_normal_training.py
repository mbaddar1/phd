import datetime
import logging
import pickle
from time import time

import torch

# Read and experiment
"""
Probability Density Function Transformation
https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables
https://www.cl.cam.ac.uk/teaching/2003/Probability/prob11.pdf 
https://en.wikibooks.org/wiki/Probability/Transformation_of_Random_Variables
https://en.wikibooks.org/wiki/Probability/Transformation_of_Probability_Densities
https://www.math.arizona.edu/~jwatkins/f-transform.pdf 
https://www.math.arizona.edu/~jwatkins/f-transform.pdf
"""
# Implementation technicalities
"""
Understanding Shapes in PyTorch
https://bochang.me/blog/posts/pytorch-distributions/ 
"""
# TODO
"""
How to generate z(t) with large number of samples (10K) with homogenous t over batches !
"""
from torch.distributions import MultivariateNormal

from cnf.training import cnf_fit, plot_distribution, generate_samples_cnf


def get_target_norm_dist(d: int) -> dict[str, MultivariateNormal]:
    # https://discuss.pytorch.org/t/sample-from-mixture-density/144645
    # parameters for Normal must be float
    mio = float(5)

    # code to get scale-tril mtx, comment it and make it fixed for re-reproducibility
    # scale_tril = torch.tril(torch.distributions.Uniform(low=0.1,high=1.5).sample((d,d)))
    scale_tril = torch.tensor([[0.1999, 0.0000, 0.0000, 0.0000],
                               [0.3015, 0.2242, 0.0000, 0.0000],
                               [1.1173, 0.3950, 0.9838, 0.0000],
                               [0.2646, 1.1769, 0.8492, 0.8105]])
    scale_tril_diag = torch.diag(torch.diag(scale_tril))

    assert scale_tril.shape[0] == d, f"scale-tril shape doesn't match d , {scale_tril.shape[0]}!= {d}"

    target_dist_diag = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril_diag)
    target_dist_tril = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril)
    return {'target_dist_diag': target_dist_diag, 'target_dist_tril': target_dist_tril}


def get_target_norm_dist(d: int) -> dict[str, MultivariateNormal]:
    # https://discuss.pytorch.org/t/sample-from-mixture-density/144645
    # parameters for Normal must be float
    mio = float(5)

    # code to get scale-tril mtx, comment it and make it fixed for re-reproducibility
    # scale_tril = torch.tril(torch.distributions.Uniform(low=0.1,high=1.5).sample((d,d)))
    scale_tril = torch.tensor([[0.1999, 0.0000, 0.0000, 0.0000],
                               [0.3015, 0.2242, 0.0000, 0.0000],
                               [1.1173, 0.3950, 0.9838, 0.0000],
                               [0.2646, 1.1769, 0.8492, 0.8105]])
    scale_tril = scale_tril[:d, :d]  # FIXME a hack
    scale_tril_diag = torch.diag(torch.diag(scale_tril))

    assert scale_tril.shape[0] == d, f"scale-tril shape doesn't match d , {scale_tril.shape[0]}!= {d}"

    target_dist_diag = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril_diag)
    target_dist_tril = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * d), scale_tril=scale_tril)
    return {'target_dist_diag': target_dist_diag, 'target_dist_tril': target_dist_tril}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # CNF-Model params
    dry_run = False
    D_z = 2  # dimension of latent variable
    t0 = 0
    t1 = 10
    hidden_dim = 64
    width = 512
    n_iters = 1 if dry_run else 500  # experimental , loss converges around that number
    train_batch_size = 1 if dry_run else 100
    test_sample_size = 10 if dry_run else 2000
    timestamp = datetime.datetime.now()
    # Start main code
    logger.info('Getting base distribution')
    base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D_z), scale_tril=torch.eye(D_z))

    sample_z0 = base_dist.sample((test_sample_size,))
    plot_distribution(x=sample_z0, filename='normal_0_I.png')
    dist_dict = get_target_norm_dist(d=D_z)

    # train N(0,1) => N(mio,diag(Sigma))
    target_dist_diag = dist_dict['target_dist_diag']
    logger.info(f'Train CNF for N(0,I) => N(mio,diag(Sigma)) with D_z = {D_z}')
    start_time = time()
    cnf_func_fit_norm_diag, _ = cnf_fit(base_dist=base_dist, target_dist=target_dist_diag, t0=t0, t1=t1,
                                        in_out_dim=D_z,
                                        hidden_dim=hidden_dim, width=width, lr=1e-3, niters=n_iters,
                                        train_batch_size=train_batch_size)
    end_time = time()
    logger.info(
        f'Finished Training CNF for N(0,I) => N(mio,diag(Sigma)) with D_z = {D_z} in {end_time - start_time} seconds')
    model_dump_filename = f'../../models/normal/cnf_fit_N_0_1_N_mio_diag_Sigma_D={D_z}_{timestamp}.pkl'
    pickle.dump(obj=cnf_func_fit_norm_diag, file=open(model_dump_filename, 'wb'))
    logger.info(f'Successfully saved model to file {model_dump_filename}')
    zT_diag_sample, _ = generate_samples_cnf(cnf_func_fit=cnf_func_fit_norm_diag, base_dist=base_dist,
                                             n_samples=test_sample_size, t0=t0, t1=t1, is_f_t_evals=True)
    plot_distribution(x=zT_diag_sample, filename=f'generated_sample_Norm_mio_diag_Sigma.png')
    logger.info(f'successfully plotted sample distribution from fitted model')
    logger.info('##################################################################################')
    ###############################################################################
    # train N(0,1) N(mio,tril(Sigma)))

    target_dist_tril = dist_dict['target_dist_tril']
    logger.info(f'Train CNF for N(0,I) => N(mio,Sigma) with D_z = {D_z}')

    start_time = time()
    cnf_func_fit_norm_tril, _ = cnf_fit(base_dist=base_dist, target_dist=target_dist_tril, t0=t0, t1=t1,
                                        in_out_dim=D_z,
                                        hidden_dim=hidden_dim, width=width, lr=1e-3, niters=n_iters,
                                        train_batch_size=train_batch_size)
    end_time = time()
    logger.info(
        f'Finished Training CNF for N(0,I) => N(mio,tril(Sigma)) with D_z = {D_z} in {end_time - start_time} seconds')
    model_dump_filename = f'../../models/normal/cnf_fit_N_0_1_N_mio_tril_Sigma_D={D_z}_{timestamp}.pkl'
    pickle.dump(obj=cnf_func_fit_norm_tril, file=open(model_dump_filename, 'wb'))
    zT_tril_sample, _ = generate_samples_cnf(cnf_func_fit=cnf_func_fit_norm_tril, base_dist=base_dist,
                                             n_samples=test_sample_size, t0=t0, t1=t1, is_f_t_evals=True)
    plot_distribution(x=zT_tril_sample, filename=f'generated_sample_Norm_mio_tril_Sigma.png')
