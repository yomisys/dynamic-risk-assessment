import requests
import json

# Specify URL of local Flask app
URL = "http://127.0.0.1:8000"

# Call each API endpoint
response1 = requests.post(
    f"{URL}/prediction"
).json()

response2 = requests.get(
    f"{URL}/scoring"
).json()

response3 = requests.get(
    f"{URL}/summarystats"
).json()

response4 = requests.get(
    f"{URL}/diagnostics"
).json()

# Combine responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

# Write responses to file
with open("apireturns.txt", "w") as file:
    json.dump(
        responses,
        file,
        indent=4
    )

print("API responses saved to apireturns.txt")