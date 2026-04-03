from typing import Tuple
import pandas as pd
from zenml import step

@step
def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_df = df.iloc[: int(len(df)*0.8)]
    X_val_df = df.iloc[int(len(df)*0.8):]
    return X_train_df, X_val_df
