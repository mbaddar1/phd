import datetime
import logging
import os.path
import pickle

import torch
from matplotlib import pyplot

from cnf.examples.cnf_gmm_train import generate_samples_cnf, plot_distribution

if __name__ == '__main__':
    LOGGING_LEVEL = logging.DEBUG
    logging.basicConfig(level=LOGGING_LEVEL)
    logger = logging.getLogger('Main')
    saved_models_path = './models'

    D = 4
    n_batches = 100
    n_samples = 50
    timestamp = ""
    t0, t1 = 0, 10
    log_f_t = True  # a hack flag to trace f_t evaluations
    model_filename = "cnf_func_fit_gmm_K_3_D_4_niters_3000_2022-04-05T10:44:04.398428.pkl"
    cnf_func_loaded = torch.load(os.path.join(saved_models_path, model_filename))
    logger.info(f'Model f{model_filename} loaded successfully')
    bast_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D),
                                                       scale_tril=torch.diag(torch.ones(D)))
    timestamp_ = datetime.datetime.now().isoformat()
    samples_dir = f"samples/samples_{timestamp_}"
    os.makedirs(samples_dir)

    is_f_t_evals = True
    logger.info(f'number of batches = {n_batches} and number of samples = {n_samples}')
    x_gen_agg = None
    for b in range(1, n_batches + 1):

        ft_dict_filename = f"ft_dict_{os.path.splitext(model_filename)[0]}_batch_{b}.pkl"

        logger.debug(f'Generating samples and derivative function trajectory for batch # {b}')

        x_gen, ft_dict = generate_samples_cnf(cnf_func_fit=cnf_func_loaded, base_dist=bast_dist, n_samples=n_samples,
                                              t0=t0, t1=t1, is_f_t_evals=is_f_t_evals)
        x_gen = torch.tensor(x_gen.detach()) # FIXME requires grad issue ??
        if x_gen_agg is None:
            x_gen_agg = x_gen
        else:
            x_gen_agg = torch.cat([x_gen_agg,x_gen],dim=0)
        logger.debug(f'Dumping pkl file for derivative function trajectory for batch {b}')
        pickle.dump(ft_dict, open(os.path.join(samples_dir, ft_dict_filename), "wb"))
        logger.debug(f'Dumping plot for generated samples for batch {b}')
    plot_file_name = f"x_gen_kde_{os.path.splitext(model_filename)[0]}.png"
    plot_distribution(x=x_gen_agg, filename=os.path.join(samples_dir, plot_file_name))
