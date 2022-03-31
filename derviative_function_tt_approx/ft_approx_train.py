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

    # Training configs
    dry_run = False
    batch_size = 10 if dry_run else 100
    n_batch = 10
    train_meta_data_file_prefix = 'train_meta_data_dz_dt'
    train_meta_data_dir = 'train_meta_data'
    timestamp_ = datetime.datetime.now().isoformat()
    n_iterations = 10 if dry_run else 100  # 2000

    n_epochs = 10
    ######
    for d_y_idx in range(D_y):
        for epoch in range(n_epochs):
            for bi in range(n_batch):  # batch index
                status = f"Status: epoch # {epoch + 1} \t batch # {bi + 1} \t dim_idx {d_y_idx + 1} n_samples " \
                         f"= {batch_size} \t ranks = {ranks}"
                print (f'{status} \t memory vmem percentage = {psutil.virtual_memory().percent}')
                random_idx = np.random.randint(low=0, high=N - 1, size=batch_size)
                X_train = X[random_idx, :].double()
                Y_train = Y[random_idx, d_y_idx].view(-1, 1).double()
                # rule = Dörfler_Adaptivity(delta=1e-6, maxranks=[10] * (order - 1), dims=phi_dims, rankincr=1)
                xTT.fit(X_train, Y_train, iterations=n_iterations, verboselevel=1, rule=None, reg_param=1e-6)
                print(f"finished batch {bi+1}")
            print(f"finished epoch {epoch+1}")
        print(f"finished dimension {d_y_idx}")
    print("Finished training nested loop")
