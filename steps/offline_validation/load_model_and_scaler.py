from typing import Tuple
from zenml import step
import joblib
import torch
import json

from steps.offline_training.trainer_ae import AutoEncoder
from steps.offline_training.trainer_lstm import LSTMModel


@step
def load_model_and_scaler(entity_id: str, model_name: str) -> Tuple[object, object]:
    base = f"artifacts/{model_name}/{entity_id}"

    # scaler（all models）
    scaler = joblib.load(f"{base}/scaler.pkl")

    if model_name == "AE":
        # AE: use config restore structure parameters
        with open(f"{base}/model_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = AutoEncoder(cfg["input_dim"], cfg["latent_dim"])
        model.load_state_dict(torch.load(f"{base}/ae_model.pt", map_location="cpu"))
        model.eval()

    elif model_name == "LSTM":
        with open(f"{base}/model_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = LSTMModel(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"]
        )
        model.load_state_dict(torch.load(f"{base}/lstm_model.pt", map_location="cpu"))
        model.eval()
        

    elif model_name == "LOF":
        model = joblib.load(f"{base}/lof_model.pkl")

    elif model_name == "IF":
        model = joblib.load(f"{base}/if_model.pkl")

    else:
        raise RuntimeError(f"Unknown model_name: {model_name}")

    return model, scaler
