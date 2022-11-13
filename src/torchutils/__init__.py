import numpy as np
import pandas as pd
import torch
from typing import Union


def pd_to_torch_float(df: Union[pd.DataFrame, pd.Series]) -> torch.Tensor:
    return torch.from_numpy(df.values).float()

def safe_cast_to_numpy(ary):
    if isinstance(ary, np.ndarray):
        return ary
    elif isinstance(ary, torch.Tensor):
        return ary.detach().cpu().numpy()
    elif hasattr(ary, 'values'):  # covers all pandas dataframe/series types
        return ary.values
    else:
        raise NotImplementedError(f"unsupported type: {type(ary)}")