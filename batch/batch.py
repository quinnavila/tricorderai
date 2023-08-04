import pandas as pd

import xgboost as xgb

import wandb
from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ClassificationQualityMetric, ClassificationConfusionMatrix


column_mapping = ColumnMapping(
    prediction='prediction',
    target='outcome'
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ClassificationQualityMetric(),
    ClassificationConfusionMatrix()
])

@task
def get_model():
    # Download the latest model from WanDB to test
    api = wandb.Api()
    artifact = api.artifact('qmavila/model-registry/mortality-prediction:latest', type='model')
    artifact.download(root="models")


    # geting model filename
    import re
    pattern = r'([^/]+\.json)$'

    match = re.search(pattern, artifact.file())

    if match:
        filename = match.group(1)
        print(filename)
    else:
        print("Filename not found.")
    return filename

@task
def load_model(filename):
    model = xgb.Booster()
    model.load_model(f'models/{filename}')
    return model

@task
def load_data():

    raw_data = pd.read_csv('data/data01.csv')
    raw_data.drop(['group','ID'],axis=1,inplace=True)
    raw_data = raw_data.dropna(subset=['outcome'])
    
    return raw_data

def transform_data(raw_dataframe):
    transformed_df = raw_dataframe.drop('outcome', axis=1)
    return transformed_df




@task
def run_report(reference_data, current_data):
    report.run(reference_data = reference_data, current_data = df,
		column_mapping=column_mapping)
    return report

@task
def run_batch(model, X):

    dmatrix = xgb.DMatrix(X)
    prediction = model.predict(dmatrix)

    return prediction


@flow
def main_batch_flow():
    # We can load data from s3 as needed in the future
    current_data = load_data()
    transformed_data = transform_data(current_data)
    
    # Using latest model from registry run predictions
    model_filename = get_model()
    model = load_model(model_filename)
    prediction = run_batch(model, transformed_data)

    current_data['prediction'] = prediction

    # cleaning up to match reference data
    current_data['prediction'] = (current_data['prediction'] >= 0.5).astype(float)

    reference_data = pd.read_parquet('data/reference.parquet')
    report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)
    
    # This can be used for
    result = report.as_dict()
    print(result.keys())



if __name__=="__main__":
    main_batch_flow()





# # Download the latest model from WanDB to test
# api = wandb.Api()
# artifact = api.artifact('qmavila/model-registry/mortality-prediction:latest', type='model')
# artifact.download(root="models")


# # geting model filename
# import re
# pattern = r'([^/]+\.json)$'

# match = re.search(pattern, artifact.file())

# if match:
#     filename = match.group(1)
#     print(filename)
# else:
#     print("Filename not found.")


# # Lets test the model downloaded model
# df = pd.read_csv('data/data01.csv')

# df.drop(['group','ID'],axis=1,inplace=True)
# df = df.dropna(subset=['outcome'])

# X = df.drop('outcome', axis=1)
# y = df['outcome']

# model = xgb.Booster()
# model.load_model(f'models/{filename}')

# dmatrix = xgb.DMatrix(X)
# prediction = model.predict(dmatrix)

# df['prediction'] = prediction

# # cleaning up to match reference data
# df['prediction'] = (df['prediction'] >= 0.5).astype(float)


