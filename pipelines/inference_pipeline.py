from zenml import pipeline
from steps import data_loader,inference_data_combined



@pipeline
def inference_pipeline(entity_id: str):
    # 1) Load TEST data
    df_test = data_loader(entity_id=entity_id, data_type="test")

    # 2) Load TEST LABEL data
    df_label = data_loader(entity_id=entity_id, data_type="test_label")

    # 3) Combine feature + label
    df_combined = inference_data_combined(
        test_df=df_test,
        label_df=df_label
    )

    # 4) Split into val / inference / deploy
    #val_df, inf_df, deploy_df = inference_data_split(df=df_combined)

    return df_combined
