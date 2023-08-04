import pytest
from fastapi.testclient import TestClient
from app.predict import app, prepare_features
import pandas as pd

client = TestClient(app)


@pytest.fixture
def mock_prepare_features(mocker):
    mock_fn = mocker.patch("app.predict.prepare_features")
    # Replace this with your mock data
    mock_fn.return_value = {'Lactic acid': 0.5, 'Heart rate': 75.0, 'Temperature': 98.6}
    return mock_fn

# Mocked predict function (if needed)
@pytest.fixture
def mock_predict(mocker):
    mock_fn = mocker.patch("app.predict.predict")
    # Replace this with your mock prediction result (0 or 1)
    mock_fn.return_value = 1
    return mock_fn

# Test your endpoint
def test_predict_endpoint(mock_prepare_features, mock_predict):
    vitals_data = {'Lactic acid': 0.5, 'Heart rate': 75.0, 'Temperature': 98.6}
    response = client.post("/predict", json=vitals_data)

    assert response.status_code == 200
    assert response.json() == {'outcome': 1}


def test_prepare_features():
    # Mock data for testing
    vitals = {
        "Lactic acid": 0.5,
        "Heart rate": 75.0,
        "Temperature": 98.6,
        "group": "A",
        "ID": 123
    }

    # Call the function to be tested
    result = prepare_features(vitals)

    # Assert the result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the columns 'group' and 'ID' are not present in the DataFrame
    assert 'group' not in result.columns
    assert 'ID' not in result.columns

    # Assert the DataFrame has the expected columns
    expected_columns = ['Lactic acid', 'Heart rate', 'Temperature']
    assert result.columns.tolist() == expected_columns

    # Assert the DataFrame has the expected values
    expected_values = pd.DataFrame([{
        'Lactic acid': 0.5,
        'Heart rate': 75.0,
        'Temperature': 98.6
    }])
    assert result.equals(expected_values)


