import os.path
import pickle
import logging

import torch


def gen_data(data_path):
    X = Y = None
    logger = logging.getLogger()
    for file in os.listdir(data_path):
        if file.endswith(".pkl"):
            logger.debug(f"Getting data from batch-part file {file}")
            ft_part = pickle.load(open(os.path.join(data_path, file), 'rb'))
            B = ft_part['z0'].size()[0]
            if Y is None:
                Y = torch.cat(ft_part['f_t'])
            else:
                Y = torch.cat([Y, torch.cat(ft_part['f_t'])])
            if X is None:
                t = torch.repeat_interleave(torch.tensor(ft_part['t']), B)
                z_t = torch.cat(ft_part['z_t'])
                X = torch.cat(tensors=[z_t, t.view(-1, 1)], dim=1)
            else:
                t = torch.repeat_interleave(torch.tensor(ft_part['t']), B)
                z_t = torch.cat(ft_part['z_t'])
                X = torch.cat([X, torch.cat(tensors=[z_t, t.view(-1, 1)], dim=1)])
            logger.debug(f'Accumulated_X shape = {X.size()}')
            logger.debug(f'Accumulated_Y shape = {Y.size()}')
    logger.debug(f'Final_X shape = {X.size()}')
    logger.debug(f'Final_Y shape = {Y.size()}')
    return X, Y


if __name__ == '__main__':
    LOGGING_LEVEL = logging.DEBUG
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=LOGGING_LEVEL, format=FORMAT)
    logger = logging.getLogger()
    samples_dir = "samples/samples_2022-03-13T10:09:04.888518"
    data_dir = 'data'

    data_version = samples_dir.split("_")[1]
    X, Y = gen_data(samples_dir)

    pickle.dump(obj={'X': X, 'Y': Y}, file=open(os.path.join(f'data/data_{data_version}.pkl'), 'wb'))
    # models_dir = "models"
    # ft_pkl_filename = "ft_dict_cnf_func_fit_gmm_K_3_D_2_niters_10_2022-03-11T10:34:27.159668.pkl"
    # ft_dict = pickle.load(open(os.path.join(models_dir, ft_pkl_filename), 'rb'))
    # logger.debug(ft_dict.keys())
