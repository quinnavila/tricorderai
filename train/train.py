import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import wandb
from wandb.xgboost import WandbCallback
from prefect import flow, task

@task
def training_data():
    df = pd.read_csv('data/data01.csv')
    df.drop(['group','ID'],axis=1,inplace=True)
    df = df.dropna(subset=['outcome'])
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

@task
def best_parameters(sweep_id="w2o5fkw8"):
    # sweep_id = "w2o5fkw8"
    api = wandb.Api()
    sweep = api.sweep(f"qmavila/mlops-mortality-prediction/sweeps/{sweep_id}")

    # Lets look at the best run
    best_run = sweep.best_run()
    best_parameters = best_run.config
    print(best_run.name)


    bst_params = {
        'objective': 'binary:logistic',
        'gamma': best_parameters['gamma'],
        'learning_rate': best_parameters['learning_rate'],
        'max_depth': best_parameters['max_depth'],
        'min_child_weight': best_parameters['min_child_weight'],
        'n_estimators': best_parameters['n_estimators'],
        'nthread': 24,
        'random_state': 42,
        'reg_alpha': best_parameters['reg_alpha'],
        'reg_lambda': best_parameters['reg_lambda'],
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist' 
            }
    return bst_params

@task
def train_best_model(X_train, X_test, y_train, y_test, bst_params):
    run = wandb.init(project="mlops-mortality-prediction", job_type='train-model')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBClassifier(
        **bst_params,
        callbacks=[WandbCallback(log_model=True)]
    )


    # Train the XGBoost model
    xgb_model.fit(X_train, y_train)

    # Make predictions on the training set
    y_train_pred = xgb_model.predict(X_train)


    # Make predictions on the test set
    y_test_pred = xgb_model.predict(X_test)

    # Calculate accuracy for training and testing sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Calculate F1 score for training and testing sets
    train_f1_score = f1_score(y_train, y_train_pred)
    test_f1_score = f1_score(y_test, y_test_pred)

    # Calculate AUC-ROC score for training and testing sets
    train_auc_roc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
    test_auc_roc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    

    # 
    # X_train['prediction'] = y_train_pred
    # X_test['prediction'] = y_test_pred
    # reference = X_test.copy()
    # reference['outcome'] = y
    # reference.to_parquet('data/reference.parquet')


    # print("Training Accuracy:", train_accuracy)
    # print("Testing Accuracy:", test_accuracy)
    # print("Training F1 Score:", train_f1_score)
    # print("Testing F1 Score:", test_f1_score)
    # print("Training AUC-ROC:", train_auc_roc)
    # print("Testing AUC-ROC:", test_auc_roc)

    run.log({
        "Train-Accuracy": train_accuracy,
        "Testing-Accuracy": test_accuracy,
        "Train-F1 Score": train_f1_score,
        "Testing-F1 Score": test_f1_score,
        "Train-AUC-ROC": train_auc_roc,
        "TestingAUC-ROC": test_auc_roc
    })
    run.finish()

@flow
def main_traing_flow():
    X_train, X_test, y_train, y_test = training_data()
    bst_params = best_parameters()
    train_best_model(X_train, X_test, y_train, y_test, bst_params)

if __name__=="__main__":
    main_training_flow()



# sweep_id = "w2o5fkw8"
# api = wandb.Api()
# sweep = api.sweep(f"qmavila/mlops-mortality-prediction/sweeps/{sweep_id}")

# # Lets look at the best run
# best_run = sweep.best_run()
# best_parameters = best_run.config
# print(best_run.name)


# bst_params = {
#     'objective': 'binary:logistic',
#     'gamma': best_parameters['gamma'],
#     'learning_rate': best_parameters['learning_rate'],
#     'max_depth': best_parameters['max_depth'],
#     'min_child_weight': best_parameters['min_child_weight'],
#     'n_estimators': best_parameters['n_estimators'],
#     'nthread': 24,
#     'random_state': 42,
#     'reg_alpha': best_parameters['reg_alpha'],
#     'reg_lambda': best_parameters['reg_lambda'],
#     'eval_metric': ['auc', 'logloss'],
#     'tree_method': 'hist' 
#         }

