from typing import Tuple, Dict
from zenml import step
import numpy as np
import torch
import time
import os

@step(enable_cache=False)
def compute_scores_block(
    X: np.ndarray,
    model,
    model_name: str,
    entity_id:str,
) -> Tuple[np.ndarray, float, float]:

    start = time.perf_counter()

    if model_name == "AE":
        # X: (N, W, F)
        N, W, F = X.shape
        X_flat = X.reshape(N, W * F)
        X_tensor = torch.tensor(X_flat, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            scores = ((preds - X_tensor) ** 2).mean(dim=1).numpy()

    elif model_name == "LSTM":
        # X: (N, W, F)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            true_last = X_tensor[:, -1, :]
            scores = ((preds - true_last) ** 2).mean(dim=1).numpy()

    elif model_name in ["IF", "LOF"]:
        # X: (N, F)
        scores = -model.decision_function(X)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    end = time.perf_counter()

    latency = (end - start) / len(scores)
    throughput = len(scores) / (end - start)
    out_dir = f"artifacts/scores/{entity_id}" 
    os.makedirs(out_dir, exist_ok=True)

# 2 define path artifacts/scores/machine-1-1/LSTM_scores.npy
    out_path = f"{out_dir}/{model_name}_scores.npy"


    np.save(out_path, scores)

    return scores, latency, throughput