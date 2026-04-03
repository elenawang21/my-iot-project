from zenml import step
from typing import List, Tuple
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from .sliding_window import sliding_window
from .minmax_scaler import fit_deep_minmax_scaler, transform_deep_with_scaler

@step
def fe_deep(df: pd.DataFrame, window_size: int, entity_id: str, model_name: str) -> np.ndarray:
    """
    Feature engineering for deep models (AE / LSTM).
    save AE scaler
    """
    feature_cols = df.columns.tolist()      
    X = sliding_window(df, window_size, feature_cols)
    scaler = fit_deep_minmax_scaler(X)
    
    
    folder = f"artifacts/{model_name}/{entity_id}"
    os.makedirs(folder, exist_ok=True)

    scaler_path = f"{folder}/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    X_scaled = transform_deep_with_scaler(X, scaler)
    return X_scaled


