"""
Memory profile refs
https://pympler.readthedocs.io/en/latest/tutorials/muppy_tutorial.html
https://pypi.org/project/memory-profiler/
"""
import datetime
import gc
import os.path
import pickle
import logging

import numpy as np
import psutil
from pympler.classtracker import ClassTracker
# from feature_utils import orthpoly_basis
# from tt import Extended_TensorTrain, Dörfler_Adaptivity
from TT.feature_utils import orthpoly_basis
from TT.tt import Extended_TensorTrain, Dörfler_Adaptivity

if __name__ == '__main__':
    print(gc.isenabled())
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
    ranks = [10] * (order - 1)

    # Feature Generation
    tfeatures = orthpoly_basis(degrees=phi_dims, domain=[-1., 1], norm='H1')
    logger.debug(f'Features Generated')

    xTT = Extended_TensorTrain(tfeatures, ranks)
    tracker = ClassTracker()
    tracker.track_object(xTT)
    logger.debug(f'Generated xTT')

    # Training
    dry_run = False
    n_samples = 10 if dry_run else 500
    random_idx = np.random.randint(low=0, high=N - 1, size=n_samples)
    epochs = 1

    train_meta_data_file_prefix = 'train_meta_data_dz_dt'
    train_meta_data_dir = 'train_meta_data'
    timestamp_ = datetime.datetime.now().isoformat()
    n_iterations = 10 if dry_run else 100  # 2000
    n_batch_iter = 10
    max_rank = 10
    # TODO
    # epochs
    # all dims
    for bi in range(n_batch_iter):  # batch index
        print(f"batch # {bi + 1} with memory percentage = {psutil.virtual_memory().percent}")
        for d_y_idx in range(D_y):
            # memory profiling
            # memory_log_filename = f"memory_profiling_logs/memory_profile_log_Yd_{d_y_idx}_max_rank_{max_rank}
            # _niter_{n_iterations}_nsample_{n_samples}.logs"
            # create_memory_profiler_logger(logger_name="memory_profile_logger", log_file_name=memory_log_filename,
            #                               logging_level=logging.DEBUG, log_format=FORMAT)
            # ##
            logger.info(f'Starting training with at epoch with N = {n_samples} for y_d|d={d_y_idx + 1} '
                        f'with max_rank = {max_rank}')
            X_train = X[random_idx, :].double()
            Y_train = Y[random_idx, d_y_idx].view(-1, 1).double()
            rule = Dörfler_Adaptivity(delta=1e-6, maxranks=[max_rank] * (order - 1), dims=phi_dims, rankincr=1)
            xTT.fit(X_train, Y_train, iterations=n_iterations, verboselevel=1, rule=None, reg_param=1e-6)
            tracker.stats.print_summary()
            train_meta_data_filepath = os.path.join(train_meta_data_dir,
                                                    f'{train_meta_data_file_prefix}_Yd_{d_y_idx + 1}_maxrank_{max_rank}_n_iter_{n_iterations}_nsampels_{n_samples}_{timestamp_}.pkl')
            pickle.dump(xTT.train_meta_data, open(train_meta_data_filepath, 'wb'))

            logger.info(f'Finished training with at with N = {n_samples} for y_d|d={d_y_idx + 1} '
                        f'with max_rank = {max_rank}')
            gc.collect()
