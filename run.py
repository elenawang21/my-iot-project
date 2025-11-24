# run_data_loader_test.py
from pipelines.access_data_pipeline import access_data_validation_pipeline

if __name__ == "__main__":
    print("Starting data_loader test run for machine-1-1.txt...")
    
    # 🚨 修正：直接在 .run() 方法中传递参数
    access_data_validation_pipeline.run(
        entity_id="machine-1-1.txt", 
        data_type="train"
    )
    
    print("Run finished. Please check ZenML UI for success status.")