import os.path
import pickle
import torch
from cnf.training import generate_samples_cnf

if __name__ == '__main__':
    D_z = 2  # dimension of latent variable
    t0 = 0
    t1 = 10
    test_sample_size = 9600

    model_path = "/home/mbaddar/Documents/mbaddar/phd/phd/models/normal"
    model_file = "cnf_fit_N_0_1_N_mio_diag_Sigma_D=2_2022-05-24 10:00:27.496828.pkl"
    model_pkl_path = os.path.join(model_path, model_file)
    base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D_z), scale_tril=torch.eye(D_z))

    ##########################

    cnf_func_fit = pickle.load(open(model_pkl_path, 'rb'))
    sample_z0 = base_dist.sample((test_sample_size,))
    zT_sample, ft = generate_samples_cnf(cnf_func_fit=cnf_func_fit, base_dist=base_dist,
                                         n_samples=test_sample_size, t0=t0, t1=t1, is_f_t_evals=True)
    pickle.dump(obj=ft, file=open(os.path.join(model_path, "trajectory_" + model_file), 'wb'))
