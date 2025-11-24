import pandas as pd
import os
from zenml.steps import step # 移除 BaseParameters 的导入
from zenml.logger import get_logger
from steps.config import DATASET_ROOT # 假设已导入

logger = get_logger(__name__)

# 🚨 修正：Step 函数直接接收参数
@step(enable_cache=False)
def data_loader(entity_id: str, data_type: str) -> pd.DataFrame: 
    """加载指定 Entity 和指定类型 (train/test/test_label) 的原始数据。"""
    
    # 路径构建
    folder_path = os.path.join(DATASET_ROOT, data_type)
    file_path = os.path.join(folder_path, entity_id) 
    
    logger.info(f"Loading data from: {file_path}")

    try:
        data = pd.read_csv(
            file_path, 
            header=None, 
            sep='\s+', 
            engine='python'
        )
    except FileNotFoundError:
        logger.error(f"File not found for entity: {entity_id} in {data_type} folder. Path: {file_path}")
        raise

    # 清理逻辑
    if data_type == 'test_label':
        data.columns = ['label']
    else:
        data.columns = [f'feature_{i}' for i in range(data.shape[1])]
        if 'feature_0' in data.columns:
            data = data.drop(columns=['feature_0'], errors='ignore') 
        
    logger.info(f"Successfully loaded {data_type} data for {entity_id} with shape {data.shape}.")
    
    return data