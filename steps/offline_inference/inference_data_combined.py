from zenml import step
import pandas as pd

@step
def inference_data_combined(test_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    df["label"] = label_df["label"].values
    return df
