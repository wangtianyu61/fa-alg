from functools import partial
import logging
import os
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Sequence, Optional

import folktables
import numpy as np
import pandas as pd
import torch
from folktables import state_list
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import src.datasets
from src.datasets import TGT, Dataset, DATASET_ROOT, DEFAULT_TRAIN_FRAC, \
    DatasetConfig, SENS_RACE, SENS_SEX, SENS_AGE
from src.torchutils import pd_to_torch_float

GERMAN = 'german'
WINE = "wine"

@dataclass
class TabularDatasetConfig(DatasetConfig):
    """Class for general dataset configs."""
    scale: bool = True
    make_dummies: bool = True
    bootstrap: bool = False
    label_encode_categorical_cols: bool = False
    is_regression: bool = True
    target_threshold: float = None
    random_state: int = None
    grouping_var: str = None
    domain_split_colname: str = None
    domain_split_values: list = field(default_factory=lambda: list())


def get_numeric_columns(df: pd.DataFrame):
    columns = []
    for c in df.columns:
        if (c != TGT) \
                and df[c].dtype != 'object' \
                and df[c].dtype != 'bool' \
                and not hasattr(df[c], 'cat'):
            columns.append(c)
    return columns


def get_categorical_columns(df: pd.DataFrame):
    columns = []
    for c in df.columns:
        if (df[c].dtypes == 'object') or (hasattr(df[c], 'cat')):
            columns.append(c)
    return columns


def bool_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if type(df[c]) == 'bool':
            df[c] = df[c].astype(int)
    return df


def replace_json_characters_in_colnames(df: pd.DataFrame) -> pd.DataFrame:
    clean_colnames = [
        re.sub("{|}|\[|\]|\(|\)|,|:\"", "", c) for c in df.columns]
    df.columns = clean_colnames
    return df


def scale_data(df_tr, df_te, df_val, cols_to_preserve: Sequence[str] = None):
    """Scale numeric train/test data columns using the *train* data statistics.
    """
    scaler = preprocessing.StandardScaler()
    columns_to_scale = set(get_numeric_columns(df_tr))
    
    unscaled_columns = (set(df_tr.columns) - set(columns_to_scale))
    if cols_to_preserve:
        columns_to_scale -= set(cols_to_preserve)
        unscaled_columns.update(cols_to_preserve)
        columns_to_scale = list(columns_to_scale)
    
    df_tr_scaled = pd.DataFrame(scaler.fit_transform(df_tr[columns_to_scale]),
                                columns=columns_to_scale)

    df_tr_out = pd.concat((df_tr_scaled, df_tr[unscaled_columns]), axis=1)

    df_te_scaled = pd.DataFrame(scaler.transform(df_te[columns_to_scale]),
                                columns=columns_to_scale)
    df_te_out = pd.concat((df_te_scaled, df_te[unscaled_columns]), axis=1)

    df_val_scaled = pd.DataFrame(scaler.transform(df_val[columns_to_scale]),
                                 columns=columns_to_scale)
    df_val_out = pd.concat((df_val_scaled, df_val[unscaled_columns]), axis=1)

    return df_tr_out, df_te_out, df_val_out


def make_dummy_cols(df_tr, df_te, df_val, drop_first=True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_train = len(df_tr)
    n_te = len(df_te)
    # Concatenate for dummy creation, so all columns are in both splits.
    df = pd.concat((df_tr, df_te, df_val))
    df_dummies = pd.get_dummies(df, drop_first=drop_first)
    return (df_dummies[:n_train],
            df_dummies[n_train:n_train + n_te],
            df_dummies[n_train + n_te:])


def replace_with_dummy(df, col, baseline_value):
    dummies = pd.get_dummies(df[col])
    dummies.drop(columns=baseline_value, inplace=True)
    dummies.columns = ['_'.join([col, x]) for x in dummies.columns]
    df.drop(columns=col, inplace=True)
    df = pd.concat((df, dummies), axis=1)
    return df


def label_encode_cols(df_tr, df_te, df_val) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], List]:
    """Label encode any categorical columns, also returning column metadata."""
    categorical_columns_meta = []
    n_train = len(df_tr)
    n_te = len(df_te)
    df = pd.concat((df_tr, df_te, df_val))
    for col in get_categorical_columns(df):
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col].values)
        categorical_columns_meta.append({
            "name": col,
            "num_categories": len(le.classes_),
            "idx": df.columns.tolist().index(col)})
    return (df[:n_train],
            df[n_train:n_train + n_te],
            df[n_train + n_te:]), \
           categorical_columns_meta


def bytes_features_to_string(df):
    """Convert all bytes features to string, in place."""
    for col, dtype in df.dtypes.items():
        if dtype == np.object:  # Only process byte object columns.
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))
    return


def dataframes_to_dataset(X, y, use_sens=False,
                          sensitive_attributes: List[str] = None):
    if use_sens:
        assert sensitive_attributes is not None
        return torch.utils.data.TensorDataset(
            pd_to_torch_float(X),
            pd_to_torch_float(y),
            pd_to_torch_float(X[sensitive_attributes]))
    else:
        return torch.utils.data.TensorDataset(
            pd_to_torch_float(X),
            pd_to_torch_float(y))


def dataframes_to_loader(X: pd.DataFrame, y: pd.DataFrame, batch_size: int,
                         shuffle=True, drop_last=False, use_sens=False,
                         sensitive_attributes: List[str] = None):
    dataset = dataframes_to_dataset(X, y, use_sens=use_sens,
                                    sensitive_attributes=sensitive_attributes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last)
    return loader


def split_by_column(df: pd.DataFrame, split_col: str, split_values,
                    drop_split_col=False) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """Split df based on whether the value of split_col is in split_values."""
    assert split_col in df.columns

    # Temporarily suppress SettingWithCopyWarning, since we aren't assigning
    # and only use the result for slicing.
    pd.options.mode.chained_assignment = None
    in_idxs = df[split_col].isin(split_values)
    df_in = df.loc[in_idxs]  # data with values in split_values
    df_notin = df.loc[~in_idxs]  # data with values not in split_values

    if drop_split_col:
        df_in.drop(columns=split_col, inplace=True)
        df_notin.drop(columns=split_col, inplace=True)
    pd.options.mode.chained_assignment = "warn"
    return df_notin, df_in

class TabularDataset(Dataset):
    """Class for loading, modifying, and working with datasets."""

    def __init__(self,
                 name: str,
                 scale: bool,
                 make_dummies: bool,
                 bootstrap: bool,
                 label_encode_categorical_cols: bool,
                 sens: List[str],
                 batch_size: int = None,
                 root_dir: str = DATASET_ROOT,
                 train_frac: float = DEFAULT_TRAIN_FRAC,
                 grouping_var=None,
                 is_regression: bool = True,
                 target_threshold: float = None,
                 subsample_size: float = 1.,
                 random_state=None,
                 domain_split_colname=None,
                 domain_split_values: Optional[List] = None):
        # Input checking
        if bootstrap and subsample_size < 1.:
            raise ValueError("Bootstrapping not compatible with downsampling.")

        self.batch_size = batch_size
        self.categorical_columns_meta = {}
        self.is_regression = is_regression
        self.target_threshold = target_threshold  # binarization threshold
        assert not (make_dummies and label_encode_categorical_cols), \
            "make dummies and label encoding cannot both be set to True."
        self.make_dummies = make_dummies
        self.label_encode_categorical_cols = label_encode_categorical_cols
        self.name = name

        # Placeholders for the data
        self.X_tr = None
        self.y_tr = None
        self.grp_tr = None
        self.X_te = None
        self.y_te = None
        self.grp_te = None
        self.X_val = None
        self.y_val = None
        self.grp_val = None
        print(root_dir)
        # Data/sampling parameters
        assert os.path.exists(root_dir)
        self.root_dir = root_dir
        self.scale = scale
        self.sens = sens
        self.bootstrap = bootstrap
        self.subsample_size = subsample_size
        self.grouping_var = grouping_var  # name of column used for grouping
        self.train_frac = train_frac
        self.random_state = random_state
        self.domain_split_colname = domain_split_colname
        self.domain_split_values = domain_split_values
        self._load()

    @property
    def d(self):
        return self.X_tr.shape[1]

    def get_dataloader(self, split="train", batch_size: int = None,
                       shuffle=True, drop_last=False,
                       use_sens=True) -> torch.utils.data.DataLoader:
        if batch_size is not None:
            logging.warning("overriding dataset batch size {} with {}".format(
                self.batch_size, batch_size))
        else:
            batch_size = self.batch_size
        if split == "train":
            return dataframes_to_loader(
                X=self.X_tr, y=self.y_tr, batch_size=batch_size,
                shuffle=shuffle, drop_last=drop_last, use_sens=use_sens,
                sensitive_attributes=self.sens)
        elif split == "test":
            return dataframes_to_loader(
                X=self.X_te, y=self.y_te, batch_size=batch_size,
                shuffle=shuffle, drop_last=drop_last, use_sens=use_sens,
                sensitive_attributes=self.sens)
        elif split == "validation":
            return dataframes_to_loader(
                X=self.X_val, y=self.y_val, batch_size=batch_size,
                shuffle=shuffle, drop_last=drop_last, use_sens=use_sens,
                sensitive_attributes=self.sens)
        else:
            raise ValueError

    def get_dataset_root_dir(self):
        return os.path.join(self.root_dir, self.name)

    @property
    def n_groups(self):
        return np.prod(self.X_tr[self.sens].nunique(axis=0).values)

    def _load(self) -> None:
        """Loads a dataset and save to X/y te/tr attributes of dataset object.
        """
        raise NotImplementedError

    def _drop_zero_variance_cols(self):
        """Drop any numeric columns with zero variance in train + test + val."""
        for col in get_numeric_columns(self.X_tr):
            if (self.X_tr[col].var() == 0) and \
                    (self.X_te[col].var() == 0) and \
                    (self.X_te[col].var() == 0):
                logging.info(f"dropping zero-variance col {col}")
                self.X_tr.drop(columns=[col], inplace=True)
                self.X_te.drop(columns=[col], inplace=True)
                self.X_val.drop(columns=[col], inplace=True)

    def _stratify(self, data):
        # return data[self.sens].agg(lambda x: ''.join(x.values.astype(str)),
        #                            axis=1).T
        if len(self.sens) == 1:
            return data[self.sens[0]]
        else:
            return data[self.sens]

    def _bootstrap_sample(self, df: Optional[pd.DataFrame] = None):
        """Take a boostrap sample of the rows from df, with same size as df.

        If df is None, returns None.
        """
        if df is None:
            return None
        n = len(df)
        return df.sample(n=n, replace=True, random_state=self.random_state)

    def _postprocess_data(self, data: pd.DataFrame = None,
                          tr: pd.DataFrame = None,
                          te: pd.DataFrame = None,
                          val: pd.DataFrame = None):
        """Performs shared dataset postprocessing
        (shuffle, split, dummy, scale, reset index)."""
        assert (data is None or (tr is None and te is None and val is None)), \
            "Provide either data (unsplit) or tr/te (pre-split), but not both."
        if self.bootstrap:
            logging.info("Taking boostrap sample of data.")
            data = self._bootstrap_sample(data) if data is not None else data
            tr = self._bootstrap_sample(data) if data is not None else tr
            te = self._bootstrap_sample(data) if data is not None else te
            val = self._bootstrap_sample(data) if data is not None else val

        # When only a single dataset is provided, we manually split it using
        # stratified sampling by sensitive attributes. Test and validation sets
        # are of equal sizes.
        if (data is not None) and (self.domain_split_colname is None):
            # Standard case; do a normal train-test split of the data.
            data = data.sample(frac=self.subsample_size,
                               random_state=self.random_state)
            tr, te_val = train_test_split(data, train_size=self.train_frac,
                                          random_state=self.random_state,
                                          stratify=self._stratify(data))
            
            # split data equally between training and validation sets
            te, val = train_test_split(te_val, train_size=0.5,
                                       random_state=self.random_state,
                                       stratify=self._stratify(te_val))
            feature_names = data.columns

        elif data is not None:
            # Case: Do a domain split of the data. Train + validation sets are
            # from the source domain; the test set is from the target domain.
            # Note that in this case, we cannot control relative size of
            # (train + validation) vs. (test) datasets, so we just keep 90%
            # of the in-domain data for training, with the remaining for val.
            logging.info(
                f"creating domain split on column {self.domain_split_colname}"
                f" with test domain values {self.domain_split_values}")
            data = data.sample(frac=self.subsample_size,
                               random_state=self.random_state)
            tr_val, te = split_by_column(
                data,
                split_col=self.domain_split_colname,
                split_values=self.domain_split_values,
                drop_split_col=True)

            tr, val = train_test_split(tr_val, train_size=0.9,
                                       random_state=self.random_state,
                                       stratify=self._stratify(tr_val))
            feature_names = tr.columns

        else:
            # Case: train/test/val data is already provided; do not downsample
            # other modify. Only perform checks to ensure identical columns.
            assert set(tr.columns) == set(te.columns)
            assert set(tr.columns) == set(val.columns)
            feature_names = tr.columns
        tr.reset_index(inplace=True, drop=True)
        te.reset_index(inplace=True, drop=True)
        val.reset_index(inplace=True, drop=True)
        self.y_tr = tr.pop(TGT)
        self.y_te = te.pop(TGT)
        self.y_val = val.pop(TGT)
        if self.grouping_var in feature_names:
            self.grp_tr = tr.pop(self.grouping_var)
            self.grp_te = te.pop(self.grouping_var)
            self.grp_val = val.pop(self.grouping_var)
        if self.scale:
            logging.info('scaling data')
            tr, te, val = scale_data(tr, te, val, cols_to_preserve=self.sens)
        if self.make_dummies:
            logging.info('creating dummy variables')
            tr, te, val = make_dummy_cols(tr, te, val)
        elif self.label_encode_categorical_cols:
            logging.info('label-encoding categorical columns')
            (tr, te, val), categorical_columns_meta = label_encode_cols(tr, te,
                                                                        val)
            self.categorical_columns_meta = categorical_columns_meta
        tr = bool_to_numeric(tr)
        te = bool_to_numeric(te)
        val = bool_to_numeric(val)
        self.X_tr = tr
        self.X_te = te
        self.X_val = val
        self._validate_data()
        if len(self.sens) == 1:
            logging.info("sensitive attribute dist (train): {}".format(
                self.X_tr[self.sens].value_counts()))
            logging.info("sensitive attribute dist (test): {}".format(
                self.X_te[self.sens].value_counts()))
        if len(self.sens) == 2:
            logging.info("sensitive attribute dist (train): {}".format(
                pd.crosstab(self.X_tr[self.sens[0]],
                            self.X_tr[self.sens[1]])))
            logging.info("sensitive attribute dist (test): {}".format(
                pd.crosstab(self.X_te[self.sens[0]],
                            self.X_te[self.sens[1]])))
        return

    def _validate_data(self):
        """Check data for null values."""
        for attr in ("X_tr", "y_tr", "X_te", "y_te", "X_val", "y_val"):
            attr_df = getattr(self, attr)
            assert attr_df is not None, f"{attr} is empty."
            assert len(attr_df), f"{attr} dataframe contains zero elements."
            null_count = pd.isnull(attr_df).values.sum()
            assert null_count == 0, \
                f"{null_count} null values in {attr} for dataset {self.name}"
        # Replace invalid JSON characters in column names
        self.X_tr = replace_json_characters_in_colnames(self.X_tr)
        self.X_te = replace_json_characters_in_colnames(self.X_te)
        self.X_val = replace_json_characters_in_colnames(self.X_val)
        if not self.is_regression:
            assert np.all(
                np.isin(self.y_tr, (0., 1.))), "nonbinary train labels."
            assert np.all(
                np.isin(self.y_te, (0., 1.))), "nonbinary test labels."
            assert np.all(
                np.isin(self.y_val, (0., 1.))), "nonbinary valid labels."
        assert np.all(np.isin(self.X_tr[self.sens].values, [0, 1])) and \
               np.all(np.isin(self.X_te[self.sens].values, [0, 1])) and \
               np.all(np.isin(self.X_val[self.sens].values, [0, 1])), \
            "expect strictly binary sensitive attributes"

        return

    def get_data(self) -> Tuple[
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series]:
        self._validate_data()
        return (self.X_tr, self.y_tr,
                self.X_te, self.y_te,
                self.X_val, self.y_val)

    def get_data_with_groups(self) -> Tuple[
        pd.DataFrame, pd.Series, pd.Series,
        pd.DataFrame, pd.Series, pd.Series,
        pd.DataFrame, pd.Series, pd.Series]:
        self._validate_data()
        return (
            self.X_tr, self.y_tr, self.grp_tr,
            self.X_te, self.y_te, self.grp_te,
            self.X_val, self.y_val, self.grp_val)



@dataclass
class AdultDatasetConfig(TabularDatasetConfig):
    sens: list = field(default_factory=lambda: [SENS_RACE, SENS_SEX])

class AdultDataset(TabularDataset):
    """The original Adult dataset."""

    def __init__(self, use_cache: bool = True, **kwargs):
        del use_cache
        super().__init__(**kwargs, name=ADULT)

    def _load(self):
        # source: https://fairmlbook.org/code/adult.html

        features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num",
                    "Martial Status",
                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain",
                    "Capital Loss",
                    "Hours per week", "Country", "Target"]
        to_categorical_features = [
            "Workclass", "Education",
            "Education-Num",
            "Martial Status",
            "Occupation", "Relationship", "Race", "Sex",
            "Country"]

        train_fp = os.path.join(self.get_dataset_root_dir(), "adult.data")
        original_train = pd.read_csv(train_fp, names=features, sep=r'\s*,\s*',
                                     engine='python', na_values="?")

        test_fp = os.path.join(self.get_dataset_root_dir(), "adult.test")
        original_test = pd.read_csv(test_fp, names=features, sep=r'\s*,\s*',
                                    engine='python', na_values="?", skiprows=1)

        num_train = len(original_train)
        original = pd.concat([original_train, original_test])
        original['Target'] = original['Target'].replace(
            {'<=50K': 0,
             '>50K': 1,
             '<=50K.': 0,
             '>50K.': 1})
        # Redundant column
        del original["Education"]

        # Code the sensitive attribute

        original['Race'] = original['Race'].apply(
            lambda x: 0 if x == "White" else 1)

        original['Sex'] = original['Sex'].apply(
            lambda x: 0 if x == "Male" else 1)

        original.rename(columns={'Target': 'target', 'Race': SENS_RACE,
                                 'Sex': SENS_SEX}, inplace=True)

        # Fix dtypes
        for colname in to_categorical_features:
            if colname in original.columns:
                original[colname] = original[colname].astype(str)

        tr = original[:num_train]
        test_val_data = original[num_train:]
        te, val = train_test_split(test_val_data, train_size=0.5,
                                   random_state=self.random_state,
                                   stratify=self._stratify(test_val_data))
        self._postprocess_data(tr=tr, te=te, val=val)
        return


@dataclass
class GermanDatasetConfig(TabularDatasetConfig):
    grouping_var: str = None
    name: str = GERMAN
    sens: list = field(default_factory=lambda: [SENS_SEX, SENS_AGE])


class GermanDataset(TabularDataset):
    """German Credit dataset.

    See https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)."""

    def __init__(self, **kwargs):
        del kwargs['use_cache']
        super().__init__(**kwargs)

    def _load(self):
        root = self.get_dataset_root_dir()
        fp = os.path.join(root, "german.data")
        df = pd.read_csv(
            fp, sep=" ", header=None)
        df.columns = ["status", "duration", "credit_history",
                      "purpose", "credit_amt", "savings_acct_bonds",
                      "present_unemployed_since", "installment_rate",
                      "per_status_sex", "other_debtors", "pres_res_since",
                      "property", "age", "other_installment", "housing",
                      "num_exist_credits", "job", "num_ppl", "has_phone",
                      "foreign_worker", TGT]
        # Code labels as in tfds; see
        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/structured/german_credit_numeric.py
        df[TGT] = 2 - df[TGT]
        # convert per_status_sex into separate columns.
        # Sens is 1 if male; else 0.
        df[SENS_SEX] = df["per_status_sex"].apply(
            lambda x: 1 if x not in ["A92", "A95"] else 0)
        # Age sens is 1 if above median age, else 0.
        median_age = df[SENS_AGE].median()
        df[SENS_AGE] = df[SENS_AGE].apply(lambda x: 1 if x > median_age else 0)
        df["single"] = df["per_status_sex"].apply(
            lambda x: 1 if x in ["A93", "A95"] else 0)
        df.drop(columns="per_status_sex", inplace=True)
        # features 15-23 are categorical/indicators
        categorical_columns = [
            "status", "credit_history",
            "purpose", "savings_acct_bonds",
            "present_unemployed_since", "single",
            "other_debtors",
            "property", "other_installment", "housing",
            "job", "has_phone",
            "foreign_worker"]
        for colname in categorical_columns:
            df[colname] = df[colname].astype('category')
        self._postprocess_data(df)




@dataclass
class WineDatasetConfig(TabularDatasetConfig):
    sens: list = field(default_factory = lambda: ['sens'])
    
    make_dummies: bool = False
    is_regression: bool = True
    scale: bool = False

class WineDataset(TabularDataset):
    def __init__(self, use_cache: bool = True, **kwargs):
        del use_cache
        super().__init__(**kwargs, name = WINE)
    def _load(self):
        root_dir = self.get_dataset_root_dir()
        
        red = self.reload(os.path.join(root_dir, "winequality-red.csv"))
        white = self.reload(os.path.join(root_dir, "winequality-white.csv"))

        # red = pd.read_csv(
        #     os.path.join(root_dir, "winequality-red.csv"), sep=";")
        # white = pd.read_csv(
        #     os.path.join(root_dir, "winequality-white.csv"), sep=";")
        red = self.reload(os.path.join(root_dir, "winequality-red.csv"))
        white = self.reload(os.path.join(root_dir, "winequality-white.csv"))
        red["red"] = 1
        white["red"] = 0
        data = pd.concat((red, white), axis=0)
        
        data.rename(columns={"red": self.sens[0], "quality": TGT},
                    inplace=True)
        
        self._postprocess_data(data)
        return

    def reload(self, path):
        column_name = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
        with io.open(path, "r", encoding="utf-8") as f:
            pack = []
            nlines = 0
            y_list = []
            for line in f:
                if nlines > 0:
                    tokens = line.strip().split(";")
                    features = np.array(tokens[0:11]).astype(float)
                    labels = int(tokens[11])
                    y_list.append(labels)
                    pack.append(features)
                nlines += 1
        X = np.array(pack)
        y = np.array(y_list).reshape(-1, 1)
        df_red = pd.DataFrame(np.hstack((X, y)), columns = column_name)
        return df_red

def get_dataset_config(dataset, **kwargs):
    if dataset == src.datasets.tabular.WINE:
        return WineDatasetConfig(**kwargs)
    elif dataset == src.datasets.tabular.ADULT:
        return AdultDatasetConfig(**kwargs)
    elif dataset == src.datasets.tabular.GERMAN:
        return GermanDatasetConfig(**kwargs)
    else:
        raise NotImplementedError