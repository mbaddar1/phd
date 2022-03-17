import os.path
import pickle

import pandas as pd
from matplotlib import pyplot
train_meta_data_dir = "train_meta_data"
plots_dir = 'plots'
train_meta_data_pkl_filename = "train_meta_data_dz_dt_Yd_1_maxrank_5_n_iter_2000_2022-03-14T16:05:20.944930.pkl"
train_meta_data_filepath = os.path.join(train_meta_data_dir, train_meta_data_pkl_filename)
train_meta_data_dict = pickle.load(open(train_meta_data_filepath, 'rb'))


series = pd.Series(train_meta_data_dict['iter_loss'])
series_ma = series[5:].rolling(window=100).mean()
series_ma[2:].plot()
plt_filename = f"plot_{train_meta_data_pkl_filename.replace('.pkl','')}.png"
pyplot.savefig(os.path.join(plots_dir,plt_filename))
print(series.describe())

