import datetime
import logging
import os.path
import pickle

import torch
from matplotlib import pyplot

from cnf.examples.cnf_gmm_train import generate_samples_cnf, plot_distribution

if __name__ == '__main__':
    LOGGING_LEVEL = logging.INFO
    logger = logging.getLogger('Main')
    saved_models_path = './models'
    K = 3
    D = 4
    n_samples = 500
    n_batches = 20
    timestamp = ""
    t0, t1 = 0, 10
    log_f_t = True  # a hack flag to trace f_t evaluations
    model_filename = "cnf_func_fit_gmm_K_3_D_4_niters_3000_2022-03-09T13:40:35.097106.pkl"
    cnf_func_loaded = torch.load(os.path.join(saved_models_path, model_filename))
    logger.info(f'Model f{model_filename} loaded successfully')
    bast_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D),
                                                       scale_tril=torch.diag(torch.ones(D)))
    timestamp_ = datetime.datetime.now().isoformat()
    out_dir = f"samples/samples_{timestamp_}"
    os.makedirs(out_dir)
    is_f_t_evals = True
    for j in range(n_batches):
        print(f'Generating batch # {j+1}')
        ft_dict_filename = f"ft_dict_{j}.pkl"
        plot_file_name = f"x_gen_kde_{j}.png"
        x_gen, ft_dict = generate_samples_cnf(cnf_func=cnf_func_loaded, base_dist=bast_dist, n_samples=n_samples,
                                              t0=t0, t1=t1, is_f_t_evals=is_f_t_evals)
        pickle.dump(ft_dict, open(os.path.join(out_dir, ft_dict_filename), "wb"))
        plot_distribution(x=x_gen, filename=os.path.join(out_dir, plot_file_name))
    pyplot.savefig('kde_gmm.png')
