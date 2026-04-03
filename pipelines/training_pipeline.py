from zenml import pipeline
from steps import data_loader, train_val_split, fe_deep, fe_deep_transform, trainer_ae, hp_tunning_ae, trainer_lstm, hp_tunning_lstm, fe_ml, fe_ml_transform, trainer_lof, hp_tunning_lof, trainer_if, hp_tunning_if

# training_pipeline.py

@pipeline(enable_cache=False)
def training_pipeline(entity_id: str, window_size: int, model_name: str):

    # 1. Load train data
    df = data_loader(entity_id=entity_id, data_type="train")
    X_train_df, X_val_df = train_val_split(df=df)

    # ---------------------------
    # 2. Deep Learning FE (AE / LSTM)
    # ---------------------------
    if model_name in ["AE", "LSTM"]:

        # fe_deep FITS scaler + saves scaler.pkl → returns only X_train_scaled
        X_train_scaled = fe_deep(
            df=X_train_df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

        # fe_deep_transform LOADS scaler.pkl → returns X_val_scaled
        X_val_scaled = fe_deep_transform(
            df=X_val_df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

        X_train_final = X_train_scaled
        X_val_final = X_val_scaled

    # ---------------------------
    # 3. ML FE (LOF / IF)
    # ---------------------------
    elif model_name in ["LOF", "IF"]:

        X_train_scaled = fe_ml(
            df=X_train_df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

        X_val_scaled = fe_ml_transform(
            df=X_val_df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

        X_train_final = X_train_scaled
        X_val_final = X_val_scaled

    else:
        raise RuntimeError(f"Unsupported model: {model_name}")

    # ---------------------------
    # 4. Tuning + Training
    # ---------------------------
    if model_name == "AE":
        best_params = hp_tunning_ae(
            X_train_scaled=X_train_final,
            X_val_scaled=X_val_final
        )
        model_path = trainer_ae(
            X=X_train_final,
            best_params=best_params,
            entity_id=entity_id
        )

    elif model_name == "LSTM":
        best_params = hp_tunning_lstm(
            X_train_scaled=X_train_final,
            X_val_scaled=X_val_final
        )
        model_path = trainer_lstm(
            X=X_train_final,
            best_params=best_params,
            entity_id=entity_id
        )

    elif model_name == "LOF":
        best_params = hp_tunning_lof(X_train_ml=X_train_final)
        model_path = trainer_lof(
            X_ml=X_train_final,
            best_params=best_params,
            entity_id=entity_id
        )

    elif model_name == "IF":
        best_params = hp_tunning_if(X_train_ml=X_train_final)
        model_path = trainer_if(
            X_ml=X_train_final,
            best_params=best_params,
            entity_id=entity_id
        )

    # ---------------------------
    # 5. Final return (NO SCALER)
    # ---------------------------
    return best_params, model_path


    

