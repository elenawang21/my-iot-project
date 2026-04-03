from zenml import pipeline
from steps import data_loader,train_val_split


@pipeline
def access_data_pipeline(entity_id: str, data_type: str): 
    
        dataset = data_loader(entity_id=entity_id, data_type=data_type)
        return dataset

    
  