# LLMs_SDOH_Integration

This repo contains our code for the paper _Large Language Models for Integrating Social Determinant of Health Data: A Case Study on Heart Failure 30-Day Readmission Prediction_.

# Requirements
Heart failure 30-day hospital readmission prediction (```HF_readmission_prediction```):
```
python 3.9
imblearn==0.0
joblib==1.2.0
numpy==1.24.4
pandas==2.0.0
pymongo==4.7.0
scikit_learn==1.4.2
shap==0.45.0
tqdm==4.65.0
xgboost==1.7.6
```

LLMs to annotate SDOH variables (```LLM_SDOH_annotation```):
```
python 3.9
datasets==2.11.0
huggingface_hub==0.17.3
numpy==1.24.4
pandas==2.0.0
peft==0.10.0
torch==2.0.0
tqdm==4.65.0
transformers==4.34.1
```


# Datasets
## Datasets

The social determinants of health (SDOH) datasets used in this study can be found below:

|   Dataset | Number of SDOH variables Used |
|---------------- | -------------- |
| [NaNDA](https://www.icpsr.umich.edu/web/ICPSR/series/1920) |  223  |
| [AHRQ SDOHD](https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html) |  506 |


# Reproducibility
## 1. LLM Experiments
For zero-shot and 1-shot inference of SDOH Domains for AHRQ and NaNDA variables, please use the commands in `LLM_SDOH_annotation/commands` folder for experiments.
For example, to perform one round of inference with the following arguments run:
```
python general_LLM_inference_rel_extraction_col_type.py --base_model='meta-llama/Llama-2-7b-chat-hf' --feat_set='a' --num_shots=0 --input_data_file='INPUT_AHRQ_tract_2010-2018.csv' --output_data_file='a_zeroshot_llama7b-chat_domain_AHRQ_outputs.csv'
```
- **Language model**: Llama-2-7b-chat-hf. **Feature set**: A (SDOH variable name), **Number of shots (inference)**: 0 (i.e., zero-shot), **Input file**: AHRQ variables, **Output file (optional)**: will be automatically named based on other arguments.

## 2. Heart Failure (HF) Readmission Prediction
The patient dataset is unavailable due to privacy reasons --- however the following commands demonstrate the steps we used to train and evaluate binary classification models (using clinical and public SDOH data):

To train binary classification models on HF 30-day hospital readmission prediction (in file, choose classification algorithm, features):
```
python bal_allfeats_nosmote_sgs_evaluate_baselines_nestKfold.py
```
To analyze results of HF models:
```
python sgs_analyze_baseline.py
```


## All Related Documents: 
- [Full list of SDOH variables used from AHRQ, NaNDA (Zenodo)](https://zenodo.org/records/10982453?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMzMwMTg5NiwiZXhwIjoxNzIyMjExMTk5fQ.eyJpZCI6ImQ2NjIxODIwLTEwODEtNDVjYi1hOWQ1LWRhOTk3YTEyM2IwYyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGQyNWNlYWFlM2Q2YTRiMTg0NDA4NzlkNTRjMDNlMCJ9.l6PwLoblL2DG9bBvFEjFzur4hVo6BgapzpOpKNqoWQeRtOIN0KFDL1cQgvW7_KAbAde0yDDTdy_SjQlllqagZg)
- [Instructions for NaNDA Variable Annotation (Zenodo)](https://zenodo.org/records/11062048?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxMzk4ODIzNywiZXhwIjoxNzIyMjExMTk5fQ.eyJpZCI6ImU3NDA5ZTZmLTNhOWQtNGE2Zi04ZWJiLTQ2OGJhMjIzYmYyMCIsImRhdGEiOnt9LCJyYW5kb20iOiJkMjY2OWY3MGZkMThmMmNkZTg4NGI0MjVhMGZkNjNmMSJ9.8KDK2L4AIyGX9zOciVUEcT9oom-WDrZ1B-MX5RJH0I9i3Im5LbZhLtSnyS7ElXUoDqDUhgAnNKiMKMWGh7gKwQ)
