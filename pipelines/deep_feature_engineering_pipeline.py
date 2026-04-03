from zenml import pipeline
from steps import data_loader,train_val_split,fe_deep

@pipeline
def deep_feature_engineering_pipeline(entity_id: str, window_size: int):
    X_train_df = data_loader(entity_id=entity_id, data_type="train")
    X_train_df, X_val_df = train_val_split(df=X_train_df)
    X_scaled, scaler = fe_deep(df=X_train_df, window_size=window_size)
    return X_scaled, scaler,X_val_df


