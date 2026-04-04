# ML-Based Trust Assessment for Inter-Organizational IIoT Data Sharing

## Overview
This project develops an ML-based anomaly detection and trust assessment framework for inter-organizational IIoT data sharing. The main goal is to translate anomaly detection results into interpretable trust scores for data consumers. The system combines anomaly rate and data consistency to assess whether shared data is reliable and usable.

The project includes both offline evaluation and an online deployment-oriented prototype. Offline experiments are used for model training validation and comparison, threshold selection, and trust score design. The online part is implemented with Streamlit and allows users to select entities, define time intervals, choose models, and inspect trust related outputs in an interactive way.

## Features
- ML-based anomaly detection on IIoT time-series data
- Comparison of multiple models, including LSTM,AE and Isolation Forest,LOF
- Offline evaluation with metrics such as precision, recall, F1-score, AUROC, latency, and throughput
- Compute model anomaly score using POT to select threshold and segment-level evaluation with Point Adjustment (PA)
- Trust score generation based on anomaly rate and consistency
- Streamlit-based online trust generator for deployment-oriented use
- ZenML-based pipeline organization for reproducible workflows

## Installation
Clone the repository and install the required dependencies.

```bash
git clone <your-repository-url>
cd <my-iot-project>
pip install -r requirements.txt
```

## Project Structure
- `artifacts/`  
  Stores saved models, models configuration, scalers, thresholds, intermediate outputs, and performance reports.

- `online_deployment/`  
  Contains the scripts used for the online trust generator.

- `pipelines/`  
  Includes ZenML pipelines for data access, feature engineering, training, and validation.

- `steps/`  
  Contains individual pipeline steps such as data loading, preprocessing, model training, hyper-tuning and evaluation.

 - `results visualisation/`  
  Contains data visualisation from results, to be specific, from sucessful/failure cases.

- `ServerMachineDataset/`  
  SMD dataset including train/test/test label/interpretation label,

- `streamlit_app.py`  
  Entry point for the Streamlit-based online trust assessment interface.

- `run.py`  
 run training pipeline or validation pipeline for different dataset.

- `main_results analysis.ipynb`  
  including metrics table, performance analysis, case study, cluster analysis.

- `consistency offline.py`  
  compute conistency using KS statistics for train/validation models, which is in offline develop stage.

- `requirements.txt`  
  Lists the required Python packages.

## Run the Online Interface

```bash
streamlit run streamlit_app.py
```

## Run the Train/Validation pipeline
run the train/validation pipeline, revise the entity name manuelly, for new training data or parameter adjustment
```bash
python run.py
```
## Trust Score Formulation

The trust score is calculated as:

```text
Trust Score = 0.6 × (1 - anomaly_rate) + 0.4 × consistency
```

Default weights:

- anomaly-rate component: `0.6`
- consistency component: `0.4`

The weights can be adjusted depending on the requirements of the data consumer.

## Models

The project evaluates multiple anomaly detection models, including:

- LSTM
- Isolation Forest
- Autoencoder
- LOF

For deployment-oriented use, LSTM and Isolation Forest are the main selected models.

## Dataset

This project uses the Server Machine Dataset (SMD) for anomaly detection experiments. Each entity is treated as an individual data provider in the trust assessment setting.
