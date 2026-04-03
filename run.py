import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.training_pipeline import training_pipeline
from pipelines.validation_pipeline import validation_pipeline


if __name__ == "__main__":
    entity = "machine-2-5"
    window = 50
    MODELS = ["IF","LSTM"]

    for model_name in MODELS:
        #print(f"\n--- Training {model_name} ---")
        #run=training_pipeline(
            #entity_id=entity,
            #window_size=window,
            #model_name=model_name
        #)
    
        
        print("\n--- Running Validation {model_name} ---")
        run = validation_pipeline(
            entity_id=entity,
            window_size=window,
            model_name=model_name
         )

    print(f"Finished validation run for model: {model_name}")