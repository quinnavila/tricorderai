
# import logging
import pandas as pd
from typing import Dict
import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import xgboost as xgb


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

app = FastAPI()


def load_model(filename):
    """Load an XGBoost model from a file.

    This function loads an XGBoost model from the specified file.

    Args:
        filename (str): The filename of the XGBoost model.

    Returns:
        xgb.Booster: The loaded XGBoost model.
    """
    model = xgb.Booster()
    model.load_model(f'{filename}')
    return model

def prepare_features(vitals):
    """
    Prepare the input features for prediction 
    by converting the vitals dictionary to a DataFrame.

    Parameters:
        vitals (Dict[str, float]): A dictionary containing vital information.

    Returns:
        pd.DataFrame: A DataFrame with the vital information.
    """
    features = pd.DataFrame([vitals])
    features.drop(['group','ID'],axis=1,inplace=True)
    return features

def predict(features):
    """
    Make a prediction using the input features.

    Parameters:
        features (pd.DataFrame): DataFrame containing the input features.

    Returns:
        int: The predicted outcome (0 or 1).
    """
    model = load_model("app/x8vxzlcm_model.json")
    dmatrix = xgb.DMatrix(features)
    preds = model.predict(dmatrix)
    preds_binary = (preds > .5).astype(int)
    return int(preds_binary[0])

@app.post("/predict", status_code=200)
def predict_endpoint(vitals: Dict[str, float]):
    """
    Endpoint to make a prediction based on the input vitals.

    Parameters:
        vitals (Dict[str, float]): A dictionary containing vital information.

    Returns:
        JSONResponse: A JSON response containing the predicted outcome.
    """
    try:
        
        features = prepare_features(vitals)
        
        pred = predict(features)

        result = {
            'outcome': pred
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Handle any errors that might occur during processing
        error_msg = {'error': str(e)}
        return JSONResponse(content=error_msg, status_code=500)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)