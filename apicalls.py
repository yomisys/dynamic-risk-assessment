import requests
import json

# URL of the running Flask application
URL = "http://127.0.0.1:8000"

############################
# Call Prediction Endpoint
############################

response1 = requests.post(
    f"{URL}/prediction",
    json={
        "filepath": "testdata/testdata.csv"
    }
).json()

############################
# Call Scoring Endpoint
############################

response2 = requests.get(
    f"{URL}/scoring"
).json()

############################
# Call Summary Statistics Endpoint
############################

response3 = requests.get(
    f"{URL}/summarystats"
).json()

############################
# Call Diagnostics Endpoint
############################

response4 = requests.get(
    f"{URL}/diagnostics"
).json()

############################
# Combine Responses
############################

responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

############################
# Write Results
############################

with open(
    "apireturns.txt",
    "w"
) as file:

    json.dump(
        responses,
        file,
        indent=4
    )

print("API responses saved to apireturns.txt")