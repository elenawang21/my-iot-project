
import os
DATASET_ROOT = "./ServerMachineDataset" 

# 训练数据路径 (用于 data_loader.py)
TRAIN_DATA_ROOT = os.path.join(DATASET_ROOT, "train") 

# 测试数据路径 (用于加载 test set)
TEST_DATA_ROOT = os.path.join(DATASET_ROOT, "test")

# 测试标签路径 (用于加载 Ground Truth)
TEST_LABEL_ROOT = os.path.join(DATASET_ROOT, "test_label")
TRAIN_VAL_SPLIT_RATIO = 0.80
TEST_PROD_SPLIT_RATIO = 0.50