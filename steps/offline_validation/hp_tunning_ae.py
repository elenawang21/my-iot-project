import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from zenml import step
from typing import Dict, Any
from typing_extensions import Annotated

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def flatten_X(X):
    N, W, F = X.shape
    return X.reshape(N, W * F)
   

def compute_reconstruction_error(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_hat = model(X_tensor).numpy()
    return np.mean((X - X_hat)**2)

def objective(trial, X_train, X_val):
    # flatten 3D → 2D
    X_train = flatten_X(X_train)
    X_val = flatten_X(X_val)

    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_categorical("epochs", [10, 20, 30])

    input_dim = X_train.shape[1]

    # model
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = criterion(out, X_train_tensor)
        loss.backward()
        optimizer.step()

    val_error = compute_reconstruction_error(model, X_val)

    return val_error

@step
def hp_tunning_ae(X_train_scaled: np.ndarray,
                  X_val_scaled: np.ndarray) -> Annotated[Dict[str, Any], "ae_best_params"]:


    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_scaled, X_val_scaled),
                   n_trials=20)

    print("AE Best params:", study.best_params)
    return study.best_params
