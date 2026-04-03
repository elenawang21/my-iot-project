from zenml import step
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from .statistics_features import statistics_features

@step
def fe_ml(
    df: pd.DataFrame,
    window_size: int,
    entity_id: str,
    model_name: str
) -> np.ndarray:
    """
    ML FE for LOF / IF.
    - compute statistical features
    - fit StandardScaler
    - save scaler inside step
    - return ONLY X_ml (no scaler return)
    """

    feature_cols = df.columns.tolist()

    # 1. Extract statistical sliding-window features
    X_df = statistics_features(df, window_size, feature_cols)

    # 2. Fit scaler
    scaler_ml = StandardScaler()
    scaler_ml.fit(X_df.values)

    # 3. Save scaler (step internal → safe)
    folder = f"artifacts/{model_name}/{entity_id}"
    os.makedirs(folder, exist_ok=True)

    scaler_path = f"{folder}/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_ml, f)

    # 4. Transform features
    X_ml = scaler_ml.transform(X_df.values)

    # 5. Return ONLY X_ml
    return X_ml
