import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fit_deep_minmax_scaler(X_3d: np.ndarray) -> MinMaxScaler:
    n, w, f = X_3d.shape
    X_2d = X_3d.reshape(-1, f)
    scaler = MinMaxScaler()
    scaler.fit(X_2d)
    return scaler

def transform_deep_with_scaler(X_3d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    n, w, f = X_3d.shape
    X_2d = X_3d.reshape(-1, f)
    X_scaled_2d = scaler.transform(X_2d)
    return X_scaled_2d.reshape(n, w, f)
