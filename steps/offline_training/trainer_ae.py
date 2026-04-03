from zenml import step
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json


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


@step
def trainer_ae(
    X: np.ndarray,
    best_params: dict,
    entity_id: str
) -> str:

    print("==== FINAL TRAIN AE ====")

    N, window_size, feature_dim = X.shape
    input_dim = window_size * feature_dim

    # flatten
    X_flat = X.reshape(N, input_dim)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # unpack hyperparameters
    latent_dim = best_params["latent_dim"]
    lr = best_params["lr"]
    epochs = best_params["epochs"]

    # build model
    model = AutoEncoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train
    for epoch in range(epochs):
        for batch in loader:
            data = batch[0]
            recon = model(data)
            loss = criterion(recon, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] loss={loss.item():.6f}")

    # save model
    model_dir = f"artifacts/AE/{entity_id}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ae_model.pt"
    torch.save(model.state_dict(), model_path)
    
    cfg_path = f"{model_dir}/model_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "window_size": window_size,
                "feature_dim": feature_dim,
            },
            f,
            indent=2
        )


    return model_path
