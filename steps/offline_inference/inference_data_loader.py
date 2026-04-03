from zenml import step
import pandas as pd
import os
from steps.config import DATASET_ROOT
from typing import Tuple

def _load_raw(entity_id: str, data_type: str) -> pd.DataFrame:
    filename = f"{entity_id}.txt"
    path = os.path.join(DATASET_ROOT, data_type, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"[inference_data_loader] File not found: {path}")

    df = pd.read_csv(path, header=None, sep=",", engine="python")
    return df

@step(enable_cache=False)
def inference_data_loader(entity_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_test = _load_raw(entity_id, "test")
    df_test.columns = [f"feature_{i}" for i in range(df_test.shape[1])]

    df_label = _load_raw(entity_id, "test_label")
    df_label.columns = ["label"]

    # optimal：force label is int（avoid afterwards etric happend with strange type）
    df_label["label"] = df_label["label"].astype(int)

    return df_test, df_label
