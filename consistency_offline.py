import os
import numpy as np
import pandas as pd

DATASET_ROOT = r"C:\Users\Yaqiong Wang\my-iot-project\ServerMachineDataset"
CSV_PATH = r"artifacts\reports\performance_metrics.csv"
target_entity = "machine-2-6"   

# read CSV
df = pd.read_csv(CSV_PATH)

# read train / test
ref = pd.read_csv(os.path.join(DATASET_ROOT, "train", f"{target_entity}.txt"),
                  header=None, sep=",", engine="python")
cur = pd.read_csv(os.path.join(DATASET_ROOT, "test", f"{target_entity}.txt"),
                  header=None, sep=",", engine="python")

# KS statistic
ks = []
for i in range(ref.shape[1]):
    x = np.sort(ref.iloc[:, i].values)
    y = np.sort(cur.iloc[:, i].values)
    z = np.sort(np.concatenate([x, y]))
    ks.append(np.max(np.abs(
        np.searchsorted(x, z, "right") / len(x) -
        np.searchsorted(y, z, "right") / len(y)
    )))

consistency = 1 - np.mean(ks)

# write in CSV
if "consistency" not in df.columns:
    df["consistency"] = None

df.loc[df["entity"].astype(str).str.strip() == target_entity, "consistency"] = consistency
df.to_csv(CSV_PATH, index=False)

print(target_entity, "consistency =", round(consistency, 4))
