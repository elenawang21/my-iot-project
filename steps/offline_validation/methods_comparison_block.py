from zenml import step
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict
import pandas as pd
import os


@step
def methods_comparison_block(
    scores: np.ndarray,
    threshold: float,
    df,
    latency: float,
    throughput: float,
    model_name: str,
    entity_id: str,
) -> Dict[str, float]:

    y_true = df["label"].values[-len(scores):]
    
    # 1. raw predict (symbole is  >)
    y_pred_raw = (scores > threshold).astype(int)
    
    # 2. Point Adjustment (PA) strategy
    y_pred = y_pred_raw.copy()
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred_raw[i] == 1:
            # once catch one anomaly point,complete the whole segment
            s, e = i, i
            while s > 0 and y_true[s-1] == 1: s -= 1
            while e < len(y_true)-1 and y_true[e+1] == 1: e += 1
            y_pred[s:e+1] = 1
    # 3. AUROC
    if len(np.unique(y_true)) < 2:
        auroc = 0.5
    else:
        # makes sure scores and anomaly is positive correlation
        auroc = roc_auc_score(y_true, scores)

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
        "threshold": float(threshold),
        "latency": latency,         # seconds / sample
        "throughput": throughput,   # samples / second
        "pred_anomaly_rate": float(np.mean(y_pred)),
    }

    print(f"\n=== {model_name} Metrics ===")
    for k, v in results.items():
        if k != "model":
            print(f"{k}: {v:.6f}")

    

    # ==========================
    # 4. APPEND TO TABLE
    # ==========================
    out_dir = "artifacts/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "performance_metric.csv")

    row = {
    "entity": entity_id,
    "model": model_name,
    "accuracy": results.get("accuracy", 0),
    "precision": results.get("precision", 0),
    "recall": results.get("recall", 0),
    "f1": results.get("f1", 0),
    "auroc": results.get("auroc", 0),
    "threshold": results["threshold"],
    "latency": latency,
    "throughput": throughput,
    "n_samples": len(y_true),
    "anomaly_rate": float(np.mean(y_true)),
    "pred_anomaly_rate": float(np.mean(y_pred)),
}

    df_row = pd.DataFrame([row])

    # first time write header
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"[methods_comparison_block] appended results -> {out_path}")
    return results



