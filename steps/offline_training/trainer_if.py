from zenml import step
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os
import json
from steps.offline_validation.threshold_selection_block import pot_threshold_upper



@step
def trainer_if(
    X_ml: np.ndarray,
    best_params: dict,
    entity_id: str
) -> str:

    print("==== Final TRAIN Isolation Forest ====")
    print("X_ml.shape =", X_ml.shape)

    # Ensure 2D input
    if len(X_ml.shape) == 3:
        X_ml = X_ml.reshape(X_ml.shape[0], -1)
        print("Flattened X_ml.shape =", X_ml.shape)

    # Build IF model from best_params
    model = IsolationForest(
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        contamination=best_params["contamination"],
        random_state=42,
    )

    model.fit(X_ml)

    # Save model
    model_path = f"artifacts/IF/{entity_id}/if_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    train_scores = -model.decision_function(X_ml)  # higher = more anomalous

    threshold = float(pot_threshold_upper(train_scores, q=0.98, level=0.95))
   

    thr_dir = f"artifacts/thresholds/{entity_id}"
    os.makedirs(thr_dir, exist_ok=True)
    thr_path = f"{thr_dir}/IF_threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(
            {"entity_id": entity_id, "model_name": "IF", "threshold": threshold},
            f,
            indent=2
        )

    print(f"[Saved] model     -> {model_path}")
    print(f"[Saved] threshold -> {thr_path} (threshold={threshold:.6f})")

    return model_path