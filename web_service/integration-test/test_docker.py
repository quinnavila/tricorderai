# pylint: disable=duplicate-code

import requests
import json


def test_predict_endpoint():
    # Read the data from 'first_row.json' and load as JSON
    with open('data/first_row.json', 'r') as f:
        payload = json.load(f)

    # Send a POST request to your predict endpoint
    url = "http://localhost:8000/predict"
    response = requests.post(url, json=payload)

    # Assert the status code is 200 (OK)
    assert response.status_code == 200

    # Assert the response content matches the expected outcome
    expected_outcome = {"outcome": 0}
    assert response.json() == expected_outcome

if __name__=="__main__":
    test_predict_endpoint()

