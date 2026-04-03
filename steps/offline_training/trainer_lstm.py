import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from zenml import step
from typing import Dict
import json
from steps.offline_validation.threshold_selection_block import pot_threshold_upper

# =============================
# 1. LSTM MODEL
# =============================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim)  # predict last step feature vector

    def forward(self, x):
        out, _ = self.lstm(x)        # out: (N, W, hidden_dim)
        last = out[:, -1, :]         # (N, hidden_dim)
        pred = self.fc(last)         # (N, input_dim)
        return pred



# =============================
# 2. TRAINING STEP
# =============================
@step
def trainer_lstm(
    X: np.ndarray,
    best_params: Dict,
    entity_id: str
) -> str:

    print("==== FINAL TRAIN LSTM ====")

    # X shape = (N, W, F)
    N, window_size, feature_dim = X.shape

    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    lr = best_params["lr"]
    epochs = best_params["epochs"]

    # Model
    model = LSTMModel(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    true_last = X_tensor[:, -1, :]   # (N, F)

    loader = DataLoader(
        TensorDataset(X_tensor, true_last),
        batch_size=32,
        shuffle=True
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] loss={loss.item():.6f}")

    # Save model
    model_dir = f"artifacts/LSTM/{entity_id}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    
    cfg_path = f"{model_dir}/model_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "window_size": window_size,
            },
            f,
            indent=2
        )
    # TRAIN SCORES -> POT THRESHOLD (save to artifacts/thresholds for online deployment)
    # =============================
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)  # (N, F)
        train_scores = ((preds - true_last) ** 2).mean(dim=1).cpu().numpy()  # (N,)

    threshold = float(pot_threshold_upper(train_scores, q=0.98, level=0.95))

    thr_dir = f"artifacts/thresholds/{entity_id}"
    os.makedirs(thr_dir, exist_ok=True)

    thr_path = f"{thr_dir}/LSTM_threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_id": entity_id,
                "model_name": "LSTM",
                "threshold": threshold
            },
            f,
            indent=2
        )

    print(f"[Saved] {model_path}")
    print(f"[Saved] {cfg_path}")
    print(f"[Saved] {thr_path} (threshold={threshold:.6f})")

    return model_path





    