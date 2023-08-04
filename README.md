# tricorderai
The tricorder is a multifunctional hand-held device that can perform environmental scans, data recording, and data analysis; hence the word "tricorder" to refer to the three functions of sensing, recording, and computing. 

This project was part of the [MLOps zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp) and uses ML to potentially show how we can better help ICU patients. While just a course project and not as powerful as the original tricorder :) a step forward.

The course covered MLOps maturity model, experiment tracking, deployment, monitoring and  best practices.

# In Hospital Mortality Prediction

## Background and Description
This project is using the kaggle in hospital mortality prediction dataset.

[In Hospital Mortality Prediction](https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction)

The predictors of in-hospital mortality for intensive care units (ICU)-admitted HF patients remain poorly characterized. We aimed to develop and validate a prediction model for all-cause in-hospital mortality among ICU-admitted HF patients.

### Project Goal
The goal from this project is to more accurately predict mortality rates for better characterization of patients. Also the MLOps course has [review criteria](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/07-project) which I followed for course completion

* Problem description
    * In order to help our patients we are using the dataset to better characterize ICU admited patients. A medical facility could potentially gather these vital and lab statistics to better characterize and treat patients. For this project we are using the Weights and Biases cloud for experiment tracking and model registry service. Prefect for workflow orchestration. Evidently for data and model monitoring. A baseline notebook is here for reference [Baseline](baseline_mortality.ipynb) The notebook briefly explores the data runs experiments with WandB and evidently reports.
* Cloud
    * Weights and Biases cloud for experiment tracking and model registry, Prefct cloud for workflow orchestration. A next step or stretch goal could be having the batch data on s3 with reporting in a cloud database. 
* Experiment tracking and model registry
    * Weights and Biases experiment tracking, hyperparamter tuning and model registry. Initial XGBoost experiment was used and a WandB sweep for hyperparameter tuning [sweep notebook](baseline_mortality.ipynb). After the model was tuned the best model was saved and linked in the WandB model registry. As a reference this check_model notebook illustrates downloading the model from the registry and using it for predictions [Model registry](check_model.ipynb)
* Workflow orchestration
    * We have basic prefect workflow orchestration using prefect cloud. The training workflow is in train/ and the batch workflow is batch/. A next step or stretch goal would potentially be setting up aws ECS tasks. 
* Model deployment
    * We have a batch and web_service. The web service is a containerized fastapi app.
* Model monitoring
    * We have evidently reports shown in our [notebook](check_model.ipynb). Also the batch run generates report metrics which can be input into a postgres grafana setup. Next steps/Strech goal to setup cloud db schema and grafana dashboards.
* Reproducibility
    * Makefile to run best model training, batch run, tests.
* Best practices
    * We have a Makfefile, linter, unitests, integration tests for the web service. 

# How to run?
## Training with prefect and WandB
In order to run and train the best run with prefect and WandB.
```bash
make run_train
```
## Batch with prefect and evidently
In order to run and train the best run with prefect and WandB.
```bash
make run_batch
```

## fastapi local dev test
```bash
make run_web
```

## fastapi build docker
```bash
make build_docker
```

## fastapi docker run
```bash
make run_docker
```

## pytest for fastapi
```bash
make test
```

## linter using ruff
```bash
make linter
```