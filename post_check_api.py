import requests

# Define the URL for the FastAPI endpoint
url = "https://render-deployment-example-6le1.onrender.com"

# Sample data for inference
sample_data = [
    {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
]

# Make the POST request to the inference endpoint
response = requests.post(url, json=sample_data)

# Check if the request was successful
if response.status_code == 200:
    # Print the predictions
    print("Predictions:", response.json())
else:
    print(f"Request failed with status code {response.status_code}")
