from zenml.pipelines import pipeline
from steps.access_data.data_loader import data_loader

# 🚨 修正：管道函数接收参数，并直接传给 Step
@pipeline(name="access_data_validation_pipeline")
def access_data_validation_pipeline(entity_id: str, data_type: str): 
    
    raw_data = data_loader(entity_id=entity_id, data_type=data_type)
    
    return raw_data