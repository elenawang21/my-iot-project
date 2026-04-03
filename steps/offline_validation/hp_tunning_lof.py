import optuna
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from zenml import step
from typing import Dict, Any
from typing_extensions import Annotated

def lof_score(X_train, n_neighbors):
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True
    )
    model.fit(X_train)
    scores = -model.decision_function(X_train)
    return scores.mean()

def objective(trial, X_train):
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    score = lof_score(X_train, n_neighbors)
    return score


@step
def hp_tunning_lof(
    X_train_ml: np.ndarray,
) -> Annotated[Dict[str, Any], "lof_best_params"]:

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train_ml), n_trials=20)

    print("LOF best:", study.best_params)
    return study.best_params
