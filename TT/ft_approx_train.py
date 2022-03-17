import datetime
import os.path
import pickle
import logging

import numpy as np

from feature_utils import orthpoly_basis
from tt import Extended_TensorTrain, Dörfler_Adaptivity

if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger()
    data_pkl_filename = "data/data_2022-03-13T10_09_04.888518.pkl"
    data_pkl = pickle.load(open(data_pkl_filename, 'rb'))
    logger.debug(f'loaded data pkl {data_pkl_filename} !')
    # load data
    X = data_pkl['X']
    Y = data_pkl['Y']
    D_x = X.size()[1]
    D_y = Y.size()[1]
    N = X.size()[0]

    # set tensor order, dimensions and ranks
    order = D_x
    phi_dims = [10] * order
    ranks = [5] * (order - 1)
    ranks2 = [1] * (order - 1)

    # Feature Generation
    tfeatures = orthpoly_basis(degrees=phi_dims, domain=[-1., 1], norm='H1')
    logger.debug(f'Features Generated')
    xTT = Extended_TensorTrain(tfeatures, ranks2)
    logger.debug(f'Generated xTT')

    # Training
    dry_run = False
    n_samples = 10 if dry_run else 1000
    random_idx = np.random.randint(low=0, high=N - 1, size=n_samples)
    epochs = 1
    max_rank_range = [2] if dry_run else [5]
    train_meta_data_file_prefix = 'train_meta_data_dz_dt'
    train_meta_data_dir = 'train_meta_data'
    timestamp_ = datetime.datetime.now().isoformat()
    n_iterations = 10 if dry_run else 2000
    for max_rank in max_rank_range:
        for epoch in range(1, epochs + 1):
            for d_y_idx in range(1):
                logger.info(f'Starting training with at epoch # {epoch} with N = {n_samples} for y_d|d={d_y_idx + 1} '
                            f'with max_rank = {max_rank}')
                X_train = X[random_idx, :].double()
                Y_train = Y[random_idx, d_y_idx].view(-1, 1).double()
                rule = Dörfler_Adaptivity(delta=1e-6, maxranks=[max_rank] * (order - 1), dims=phi_dims, rankincr=1)

                xTT.fit(X_train, Y_train, iterations=n_iterations, verboselevel=1, rule=rule, reg_param=1e-6)
                train_meta_data_filepath = os.path.join(train_meta_data_dir,
                                                        f'{train_meta_data_file_prefix}_Yd_{d_y_idx + 1}_maxrank_{max_rank}_n_iter_{n_iterations}_{timestamp_}.pkl')
                pickle.dump(xTT.train_meta_data, open(train_meta_data_filepath, 'wb'))

                logger.info(f'Finished training with at epoch # {epoch} with N = {n_samples} for y_d|d={d_y_idx + 1} '
                            f'with max_rank = {max_rank}')

"""
Train Meta Data Table
train_meta_data_sum_sin_xj_pow_2_n_iter_2000_2022-03-14T16:13:43.745088.pkl         with relative quad norm 
train_meta_data_dz_dt_Yd_1_maxrank_5_n_iter_2000_2022-03-15T11:10:51.495338.pkl     with quad-norm, no relative

"""