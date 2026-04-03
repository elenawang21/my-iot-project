from zenml import pipeline
from steps import data_loader,train_val_split,fe_ml


@pipeline
def ml_feature_engineering_pipeline(entity_id: str, window_size: int):
    X_train_df = data_loader(entity_id=entity_id, data_type="train")
    X_train_df, X_val_df = train_val_split(df=X_train_df)
    X_ml, scaler_ml = fe_ml(df=X_train_df, window_size=window_size)
    return X_ml, scaler_ml,X_val_df

