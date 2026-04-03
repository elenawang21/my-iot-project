from typing import Tuple, Dict
from zenml import pipeline
from steps import inference_data_loader, inference_data_combined, load_model_and_scaler, fe_deep_transform, fe_ml_transform, compute_scores_block, threshold_selection_block, methods_comparison_block
@pipeline
def validation_pipeline(
    entity_id: str,
    window_size: int,
    model_name: str
) -> Tuple[Dict, Dict, Dict]:
    # 1. load
    test_df, label_df = inference_data_loader(entity_id)

    # combine test + label
    df = inference_data_combined(test_df, label_df)

       

 # ---------------------------
    # 2. Feature Engineering (align with train)
    # ---------------------------
    if model_name in ["AE", "LSTM"]:

        X_val_final = fe_deep_transform(
            df=df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

    elif model_name in ["LOF", "IF"]:

        X_val_final = fe_ml_transform(
            df=df,
            window_size=window_size,
            entity_id=entity_id,
            model_name=model_name
        )

    else:
        raise RuntimeError(f"Unsupported model: {model_name}")

    # ---------------------------
    # 3. Load model
    # ---------------------------
    model, _ = load_model_and_scaler(entity_id, model_name)

    # ---------------------------
    # 4. Compute scores
    # ---------------------------
    scores, latency, throughput = compute_scores_block(
        X=X_val_final,
        model=model,
        model_name=model_name,
        entity_id=entity_id,
    )

    # ---------------------------
    # 5. Threshold + Metrics
    # ---------------------------
    threshold = threshold_selection_block(
    entity_id=entity_id,
    scores=scores,
    model_name=model_name
    )

    metrics = methods_comparison_block(
    scores=scores,
    threshold=threshold,
    df=df,
    latency=latency,
    throughput=throughput,
    model_name=model_name,
    entity_id=entity_id,
    )

    return threshold, metrics