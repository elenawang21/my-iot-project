import pandas as pd
from zenml import step
from typing_extensions import Annotated
@step
def data_check(data: pd.DataFrame) -> Annotated[pd.DataFrame, "clean_data"]:
    print("Data Check Started")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print("Missing values:\n", data.isnull().sum())
    print("Duplicated rows:", data.duplicated().sum())
    
    if "faultNumber" in data.columns:
        print("Fault class distribution:\n", data["faultNumber"].value_counts())

    return data