import logging
import os.path
import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    cnf_trajectory_path = '/home/mbaddar/Documents/mbaddar/phd/phd/models/normal'
    cnf_trajectory_file = 'trajectory_cnf_fit_N_0_1_N_mio_diag_Sigma_D=2_2022-05-24 10:00:27.496828.pkl'

    cnf_trajectory_dict = pickle.load(open(os.path.join(cnf_trajectory_path, cnf_trajectory_file), 'rb'))
    print(cnf_trajectory_dict.keys())

    t_arr = np.array(cnf_trajectory_dict['t'])
    logger.info(f't_arr stats : \n {pd.Series(t_arr).describe()}')
    for t_idx, t in enumerate(t_arr):
        pass

