from dataclasses import asdict

from src.datasets import DatasetConfig, tabular
import numpy as np
import pandas as pd
import gzip
import os
import scipy.sparse as sp


# encode the original data into vw format
def save_vw_dataset(X, y, name, ds_dir):
    # change a bit as we change the data structure of y
    n_classes = max(y) + 1 
    fname = '{}_{}'.format(name, n_classes)
    #add it to the csv path for the use of bypassing the monster
    with gzip.open(os.path.join(ds_dir, fname + '.vw.gz'), 'w') as f:
        #if the matrix is sparse then we use the nonzero value to fill.
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(list(X.loc[i].index), list(X.loc[i]))))
                f.write(str_output.encode())
        else:
            for i in range(X.shape[0]):
                str_output = '{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X.loc[i]) if val != 0))
                f.write(str_output.encode())
    
    X['class'] = pd.Series(y)
    X.to_csv(os.path.join(ds_dir, fname + '.csv'))


def get_dataset(config: DatasetConfig):
    if isinstance(config, tabular.WineDatasetConfig):
        return tabular.WineDataset(**asdict(config))
    elif isinstance(config, tabular.AdultDatasetConfig):
        return tabular.AdultDataset(**asdict(config))
    elif isinstance(config, tabular.CompasDatasetConfig):
        return tabular.CompasDataset(**asdict(config))
    elif isinstance(config, tabular.GermanDatasetConfig):
        return tabular.GermanDataset(**asdict(config))
    elif isinstance(config, tabular.CommunitiesAndCrimeDatasetConfig):
        return tabular.CommunitiesAndCrimeDataset(**asdict(config))
    elif isinstance(config, tabular.BRFSSDatasetConfig):
        return tabular.BRFSSDataset(**asdict(config))
    else:
        raise NotImplementedError