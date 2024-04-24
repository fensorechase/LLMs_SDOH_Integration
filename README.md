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



## LLM Experiments
For zero-shot and 1-shot inference of SDOH Domains for AHRQ and NaNDA variables, please use the commands in `LLM_SDOH_annotation/commands` folder for experiments.
