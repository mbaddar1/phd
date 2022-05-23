import os.path
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot

if __name__ == '__main__':
    train_meta_data_pkl_filename = "train_meta_data_dz_dt_y_2_max_rank_10_2022-05-10T14:28:57.021454_batch_size_10000_batch_idx_1_epoch_1_niter_1000.pkl"
    train_meta_data_dir = 'train_meta_data'
    train_meta_data_path = os.path.join(train_meta_data_dir, train_meta_data_pkl_filename)
    train_meta_data_file = open(train_meta_data_path, 'rb')
    plots_dir = 'plots'
    train_meta_data_pkl = pickle.load(train_meta_data_file)
    loss_iter = train_meta_data_pkl['curr_res']
    pyplot.title("Converge of loss over iterations")
    pyplot.ylabel("loss")
    pyplot.xlabel("iter-idx")
    q_ = np.quantile(a=loss_iter,q=0.99)
    loss_iter = [min(v,q_) for v in loss_iter]
    pd.Series(loss_iter).rolling(window=10).mean().plot()
    pyplot.savefig(os.path.join(plots_dir, "convergence2.png"))
