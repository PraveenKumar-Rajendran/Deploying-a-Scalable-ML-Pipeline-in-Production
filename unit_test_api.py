import pytest
from fastapi.testclient import TestClient
from main import app  # Ensure this imports your FastAPI app
import json

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Model API Created with FastAPI!"}

def test_post_predict_1():
    sample_data = {
            "age": 49,
            "workclass": "Private",
            "fnlgt": 160187,
            "education": "9th",
            "education_num": 5,
            "marital_status": "Married-spouse-absent",
            "occupation": "Other-service",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 16,
            "native_country": "Jamaica"
        }
        
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["predictions"] == ["<=50K"]

def test_post_predict_2():
    sample_data = {
            "age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States"
        }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["predictions"] == [">50K"]
