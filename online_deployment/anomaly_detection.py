import numpy as np
import torch

def make_windows(X: np.ndarray, w: int, stride: int = 1) -> np.ndarray:
    T, F = X.shape
    if T < w:
        return np.empty((0, w, F), dtype=float)
    idx = range(0, T - w + 1, stride)
    return np.stack([X[i:i+w] for i in idx], axis=0)


def anomaly_rate_lstm(X_seg: np.ndarray, lstm_model, scaler, window_size: int, threshold: float) -> float:
    if scaler is None:
        raise RuntimeError("LSTM scaler is REQUIRED but None was provided")

    X = np.asarray(X_seg)
    X = scaler.transform(X)   # ✅ 强制 scale（对齐训练）

    wins = make_windows(X, w=window_size, stride=1)
    if wins.shape[0] == 0:
        raise ValueError("Segment too short for LSTM window_size.")

    Xt = torch.tensor(wins, dtype=torch.float32)
    true_last = Xt[:, -1, :]

    with torch.no_grad():
        pred_last = lstm_model(Xt)
        scores = ((pred_last - true_last) ** 2).mean(dim=1).cpu().numpy()

    return float((scores > threshold).mean())


def anomaly_rate_if(
    X_seg: np.ndarray,
    if_model,
    scaler,
    threshold: float,
    stride: int = 1,
) -> float:
    if scaler is None:
        raise RuntimeError("IF scaler is REQUIRED but None was provided")

    X = np.asarray(X_seg, dtype=float)

    # 允许传进来的是 (T, F) 或者已经展平过的 (N, expected)
    expected = int(getattr(if_model, "n_features_in_", 0))
    if expected <= 0:
        raise RuntimeError("IF model missing n_features_in_ (model not fitted?)")

    if X.ndim != 2:
        raise ValueError(f"X_seg must be 2D, got shape {X.shape}")

    T, F = X.shape

    # ✅ case A: X already flattened to expected features (N, expected)
    if F == expected:
        X_flat = X
    else:
        # ✅ case B: window + flatten to expected
        if expected % F != 0:
            raise RuntimeError(
                f"IF expected features {expected} not divisible by input F={F}. "
                "Your offline and online feature construction are not aligned."
            )
        w = expected // F  # window size used in training
        n_win = (T - w) // stride + 1
        if n_win <= 0:
            return 0.0

        # build windows -> (n_win, w*F)
        X_flat = np.stack(
            [X[i:i+w].reshape(-1) for i in range(0, T - w + 1, stride)],
            axis=0
        )

    # ✅ scaler must match the flattened dimension
    X_flat_scaled = scaler.transform(X_flat)

    # ✅ score direction: higher = more anomalous (align with your offline choice)
    scores = -if_model.decision_function(X_flat_scaled)

    return float((scores > threshold).mean())