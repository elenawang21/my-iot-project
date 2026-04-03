import os
import numpy as np
import pandas as pd

DATASET_ROOT = r"C:\Users\Yaqiong Wang\my-iot-project\ServerMachineDataset"

_REF_CACHE = {}

def load_ref_df(entity: str) -> pd.DataFrame:
    
    if entity in _REF_CACHE:
        return _REF_CACHE[entity]

    ref = pd.read_csv(
        os.path.join(DATASET_ROOT, "train", f"{entity}.txt"),
        header=None,
        sep=",",               
        engine="python"
    )
    _REF_CACHE[entity] = ref
    return ref


def ks_consistency_same_as_offline(ref: pd.DataFrame, cur: pd.DataFrame) -> float:
    """✅ 100%复制你 offline 的 KS 逻辑"""
    ks = []
    for i in range(ref.shape[1]):
        x = np.sort(ref.iloc[:, i].values)
        y = np.sort(cur.iloc[:, i].values)
        z = np.sort(np.concatenate([x, y]))
        ks.append(np.max(np.abs(
            np.searchsorted(x, z, "right") / len(x) -
            np.searchsorted(y, z, "right") / len(y)
        )))
    return float(1 - np.mean(ks))


def online_consistency(entity: str, X_buf: np.ndarray) -> float:
    """
    entity: machine-2-6
    X_buf: (T, F) numpy
    """
    ref = load_ref_df(entity)
    cur = pd.DataFrame(X_buf)   
    return ks_consistency_same_as_offline(ref, cur)
