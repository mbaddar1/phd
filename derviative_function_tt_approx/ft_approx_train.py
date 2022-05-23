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
import torch
from pympler.classtracker import ClassTracker
import random

random.seed(1000)
# from feature_utils import orthpoly_basis
# from tt import Extended_TensorTrain, Dörfler_Adaptivity
from TT.feature_utils import orthpoly_basis, lb
from TT.tt import Extended_TensorTrain, Dörfler_Adaptivity


def get_XY(data_type, **kwargs):
    if data_type == 'dz_dt':
        # load data
        # FIXME a hack to create new tensor with requires_grad = False
        # the memory leak issue was due to requires_grad = True in input data tensor which propagated thru calculations
        # c = torch.tensor(c.detach())
        file_path = kwargs['file_path']
        data_pkl = pickle.load(open(file_path, 'rb'))
        X = torch.tensor(data_pkl['X'].detach())  # [z,t]
        Y = torch.tensor(data_pkl['Y'].detach())  # dz/dt
        return X, Y
    elif data_type == 'toy':
        dim = kwargs['dim']
        order = kwargs['order']
        max_rank = kwargs['max_rank']
        f = kwargs['f']
        N = 1000 * order * dim * max_rank ** 2
        X = lb.random.rand(N, dim) * 2 - 1
        Y = f(X).reshape(N, 1)
        return X, Y


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger()
    data_type = 'dz_dt'
    if data_type == 'dz_dt':
        data_pkl_filename = "data/data_2022-03-13T10_09_04.888518.pkl"
        X, Y = get_XY(data_type=data_type, file_path=data_pkl_filename)
    elif data_type == 'toy':
        toy_dim = 10
        f = lambda x: lb.tensor(np.tanh(np.cos(np.sum(x[:, i] ** 2 for i in range(toy_dim)))))
        X, Y = get_XY(data_type=data_type, dim=10, order=5, max_rank=5, f=f)
    else:
        raise ValueError(f'undefined data-type {data_type}')
    D_x = X.size()[1]
    D_y = Y.size()[1]
    N = X.size()[0]

    # set TT-hyperparams
    order = D_x
    phi_degree = [10] * order
    ranks = [10] * (order - 1)

    # Feature Generation
    tfeatures = orthpoly_basis(degrees=phi_degree, domain=[-1., 1], norm='H1')
    logger.debug(f'Features Generated')

    xTT = Extended_TensorTrain(tfeatures, ranks)
    tracker = ClassTracker()
    tracker.track_object(xTT)
    logger.debug(f'Generated xTT')

    # Training configs
    dry_run = False
    batch_size = 10 if dry_run else 10000
    n_batch = 1

    train_meta_data_dir = 'train_meta_data'
    timestamp_ = datetime.datetime.now().isoformat()
    n_iterations = 10 if dry_run else 1000  # 2000

    n_epochs = 1
    ######
    for dim_y_idx in range(D_y):
        for epoch in range(n_epochs):
            for bi in range(n_batch):  # batch index
                status = f"Status: epoch # {epoch + 1} \t batch # {bi + 1} \t dim_idx {dim_y_idx + 1} n_samples " \
                         f"= {batch_size} \t ranks = {ranks}"
                print(f'{status} \t memory vmem percentage = {psutil.virtual_memory().percent}')
                random_idx = np.random.randint(low=0, high=N - 1, size=batch_size)
                X_train = X[random_idx, :].double()
                Y_train = Y[random_idx, dim_y_idx].view(-1, 1).double()
                train_meta_data = xTT.fit(X_train, Y_train, iterations=n_iterations, verboselevel=1, rule=None,
                                          reg_param=1e-6)
                train_meta_data_file_name = f"train_meta_data_dz_dt_y_" \
                                            f"{dim_y_idx + 1}_max_rank_{max(ranks)}_{datetime.datetime.now().isoformat()}_" \
                                            f"batch_size_{batch_size}_batch_idx_{bi + 1}_epoch_{epoch + 1}_niter_{n_iterations}.pkl"
                pickle.dump(train_meta_data, open(f'train_meta_data/{train_meta_data_file_name}', 'wb'))
                print(f"finished batch {bi + 1}")
            print(f"finished epoch {epoch + 1}")
        print(f"finished dimension {dim_y_idx}")
    print("Finished training nested loop")
