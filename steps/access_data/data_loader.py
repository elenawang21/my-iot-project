import pandas as pd
import os
from zenml import step 
from steps.config import DATASET_ROOT
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def data_loader(entity_id: str, data_type: str) -> pd.DataFrame: 
    """加载指定 Entity 和指定类型 (train/test/test_label) 的原始数据。"""

   
    filename = f"{entity_id}.txt"

    
    file_path = os.path.join(DATASET_ROOT, data_type, filename)
    
    logger.info(f"[data_loader] Loading data from: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"[data_loader] File not found: {file_path}")
        raise FileNotFoundError(f"Missing file: {file_path}")

     
    data = pd.read_csv(
    file_path,
    header=None,
    sep=None,    
    engine="python"
)


    # 处理 test_label
    if data_type == 'test_label':
        data.columns = ['label']
    else:
        # 为 feature 命名
        data.columns = [f'feature_{i}' for i in range(data.shape[1])]
        

    logger.info(f"[data_loader] Loaded shape: {data.shape}")
    return data

    