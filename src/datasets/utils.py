from dataclasses import asdict

from src.datasets import DatasetConfig, tabular


def get_dataset(config: DatasetConfig):
    if isinstance(config, tabular.WineDatasetConfig):
        return tabular.WineDataset(**asdict(config))
    elif isinstance(config, tabular.AdultDatasetConfig):
        return tabular.AdultDataset(**asdict(config))
    elif isinstance(config, tabular.GermanDatasetConfig):
        return tabular.GermanDataset(**asdict(config))
    else:
        raise NotImplementedError