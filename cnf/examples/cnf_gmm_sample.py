import logging
import os.path

import seaborn
import torch
from matplotlib import pyplot

from cnf.examples.cnf_gmm_train import generate_samples_cnf

if __name__ == '__main__':
    LOGGING_LEVEL = logging.INFO
    logger = logging.getLogger('Main')
    saved_models_path = './models'
    K = 3
    D = 4
    n_samples = 20
    t0, t1 = 0, 10
    log_f_t = True  # a hack flag to trace f_t evaluations
    model_filename = f'cnf_func_fit_gmm_K_{K}_D_{D}.pkl'
    cnf_func_loaded = torch.load(os.path.join(saved_models_path, model_filename))
    logger.info(f'Model f{model_filename} loaded successfully')
    bast_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D),
                                                       scale_tril=torch.diag(torch.ones(D)))
    is_f_t_evals = True
    X_gen = generate_samples_cnf(cnf_func=cnf_func_loaded, base_dist=bast_dist, n_samples=n_samples, t0=t0, t1=t1,
                                 is_f_t_evals=is_f_t_evals)
    seaborn.kdeplot(X_gen.detach().numpy().reshape(-1))
    pyplot.savefig('kde.png')
