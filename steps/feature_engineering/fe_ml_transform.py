from zenml import step
import pandas as pd
import numpy as np
import pickle
import os
from .statistics_features import statistics_features

@step
def fe_ml_transform(
    df: pd.DataFrame,
    window_size: int,
    entity_id: str,
    model_name: str
) -> np.ndarray:
    """
    ML FE TRANSFORM-ONLY STEP
    Loads the saved scaler.pkl from fe_ml step.
    """
     # drop label
    df= df.drop(columns=["label"], errors="ignore")
    
    # 1. Load saved scaler
    scaler_path = f"artifacts/{model_name}/{entity_id}/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[fe_ml_transform] Scaler not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler_ml = pickle.load(f)

    # 2. Compute statistical features
    feature_cols = df.columns.tolist()
    X_df = statistics_features(df, window_size, feature_cols)

    # 3. Transform using saved scaler
    X_ml = scaler_ml.transform(X_df.values)

    return X_ml
