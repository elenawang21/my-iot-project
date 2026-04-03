import os
import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

DATASET_ROOT = r"C:\Users\Yaqiong Wang\my-iot-project\ServerMachineDataset"

_TEST_CACHE: Dict[str, np.ndarray] = {}
_REF_CACHE: Dict[str, pd.DataFrame] = {}

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def list_entities_from_test_folder() -> list[str]:
    test_dir = os.path.join(DATASET_ROOT, "test")
    files = [f for f in os.listdir(test_dir) if f.endswith(".txt")]
    ents = sorted([f.replace(".txt", "") for f in files])

    # after machine-2-7
    def parse(e: str):
        # "machine-2-7" -> (2,7)
        a, b = e.replace("machine-", "").split("-")
        return int(a), int(b)

    ents = sorted(ents, key=parse)

    return [e for e in ents if parse(e) >= (1, 1)]

def load_test_array(entity: str) -> np.ndarray:
    entity = str(entity).strip()
    if entity in _TEST_CACHE:
        return _TEST_CACHE[entity]
    p = os.path.join(DATASET_ROOT, "test", f"{entity}.txt")
    X = pd.read_csv(p, header=None, sep=",", engine="python").values.astype(float)
    _TEST_CACHE[entity] = X
    return X

def load_ref_df(entity: str) -> pd.DataFrame:
    entity = str(entity).strip()
    if entity in _REF_CACHE:
        return _REF_CACHE[entity]
    p = os.path.join(DATASET_ROOT, "train", f"{entity}.txt")
    ref = pd.read_csv(p, header=None, sep=",", engine="python")
    _REF_CACHE[entity] = ref
    return ref

def load_threshold(entity: str, model_name: str) -> float:
    p = f"artifacts/thresholds/{entity}/{model_name}_threshold.json"
    with open(p, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])

def load_lstm(entity: str):
    cfg_path = f"artifacts/LSTM/{entity}/model_config.json"
    model_path = f"artifacts/LSTM/{entity}/lstm_model.pt"
    scaler_path= f"artifacts/LSTM/{entity}/scaler.pkl"
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = LSTMModel(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"])
    )
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    scaler=joblib.load(scaler_path)
    window_size = int(cfg["window_size"])
    thr = load_threshold(entity, "LSTM")
    return model, scaler, window_size, thr

def load_if(entity: str):
    model_path = f"artifacts/IF/{entity}/if_model.pkl"
    scaler_path= f"artifacts/IF/{entity}/scaler.pkl"
    model = joblib.load(model_path)
    scaler=joblib.load(scaler_path)
    thr = load_threshold(entity, "IF")
    return model, scaler, thr
