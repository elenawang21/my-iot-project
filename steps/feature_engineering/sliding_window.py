import numpy as np
import pandas as pd
from typing import List

def sliding_window(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: List[str],
) -> np.ndarray:
    data = df[feature_cols].values
    n_samples = data.shape[0] - window_size + 1
    if n_samples <= 0:
        raise ValueError("window_size larger than number of rows in df.")

    X = np.zeros((n_samples, window_size, len(feature_cols)))
    for i in range(n_samples):
        X[i] = data[i : i + window_size]

    return X
