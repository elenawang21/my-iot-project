import optuna
import numpy as np
from sklearn.ensemble import IsolationForest
from zenml import step
from typing import Dict, Any
from typing_extensions import Annotated

def if_score(X_train, n_estimators, max_samples, contamination):
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)
    scores = -model.decision_function(X_train)
    return scores.mean()

def objective(trial, X_train):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_samples = trial.suggest_float("max_samples", 0.3, 1.0)
    contamination = trial.suggest_float("contamination", 0.001, 0.05)

    return if_score(X_train, n_estimators, max_samples, contamination)

@step
def hp_tunning_if(X_train_ml: np.ndarray) -> Annotated[Dict[str, Any], "if_best_params"]:

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train_ml), n_trials=20)

    print("IF best:", study.best_params)
    return study.best_params
