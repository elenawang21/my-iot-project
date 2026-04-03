import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from zenml import step
from typing import Dict, Any
from typing_extensions import Annotated

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim)  # predict last step

    def forward(self, x):
        out, _ = self.lstm(x)       # out: (N, W, H)
        last = out[:, -1, :]        # last hidden state (N, H)
        pred = self.fc(last)        # (N, input_dim)
        return pred


def compute_reconstruction_error_lstm(model, X):
    """
    Compare predicted last step with true last step.
    X shape = (N, W, F)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy()     # (N, F)
        true_last = X[:, -1, :]             # (N, F)

    return np.mean((true_last - preds) ** 2)

def objective(trial, X_train, X_val):
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [1, 2])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_categorical("epochs", [10, 20, 30])

    _, window, feature_dim = X_train.shape

    model = LSTMModel(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    # training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)         # (N, F)
        true_last = X_train_tensor[:, -1, :]  # (N, F)
        loss = criterion(preds, true_last)
        loss.backward()
        optimizer.step()

    # compute val error
    val_error = compute_reconstruction_error_lstm(model, X_val)

    return val_error

@step
def hp_tunning_lstm(
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray
) -> Annotated[Dict[str, Any], "lstm_best_params"]:

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, X_val_scaled),
        n_trials=20
    )

    print("LSTM Best params:", study.best_params)
    return study.best_params
