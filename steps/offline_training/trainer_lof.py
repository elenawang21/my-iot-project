from zenml import step
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import joblib
import os


@step
def trainer_lof(
    X_ml: np.ndarray,
    best_params: dict,
    entity_id: str
) -> str:

    print("==== Final TRAIN LOF ====")
    print("X_ml.shape =", X_ml.shape)

    # LOF must use 2D input (N, features)
    if len(X_ml.shape) != 2:
        raise ValueError(f"LOF expects 2D features, but got shape: {X_ml.shape}")

    # build model（use Best Params）
    lof = LocalOutlierFactor(
        n_neighbors=best_params["n_neighbors"],
        novelty=True
    )

    # train
    lof.fit(X_ml)

    # save model
    model_dir = f"artifacts/LOF/{entity_id}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/lof_model.pkl"

    joblib.dump(lof, model_path)

    print("Saved LOF model to:", model_path)
    return model_path

