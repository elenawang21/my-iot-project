import pandas as pd
from typing import List

def statistics_features(df, window_size, feature_cols):
    rolled = df[feature_cols].rolling(window_size)

    means = rolled.mean().iloc[window_size-1:]
    stds  = rolled.std().iloc[window_size-1:]
    mins  = rolled.min().iloc[window_size-1:]
    maxs  = rolled.max().iloc[window_size-1:]
    skews = rolled.skew().iloc[window_size-1:]
    kurts = rolled.kurt().iloc[window_size-1:]

    result = pd.concat(
        [means, stds, mins, maxs, skews, kurts],
        axis=1
    )

    result.columns = [
        f"{col}_{stat}"
        for col in feature_cols
        for stat in ["mean","std","min","max","skew","kurt"]
    ]

    return result.reset_index(drop=True)

