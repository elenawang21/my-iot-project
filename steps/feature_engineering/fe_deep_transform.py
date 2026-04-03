from zenml import step
import pandas as pd
import numpy as np
import pickle
import os

from .sliding_window import sliding_window
from .minmax_scaler import transform_deep_with_scaler

@step
def fe_deep_transform(df: pd.DataFrame, window_size: int, entity_id: str, model_name: str) -> np.ndarray:
    """
    TRANSFORM ONLY — load scaler from file saved in fe_deep.
    """

     #  1. drop label
    df = df.drop(columns=["label"], errors="ignore")
    
    # 1. Load saved scaler
    scaler_path = f"artifacts/{model_name}/{entity_id}/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 2. Slide window
    feature_cols = df.columns.tolist()
    X = sliding_window(df, window_size, feature_cols)

    # 3. Transform
    X_scaled = transform_deep_with_scaler(X, scaler)

    return X_scaled
