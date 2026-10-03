import os
import json
import shutil
import subprocess

import ingestion
import training
import scoring
import deployment
import reporting


############################
# Load configuration
############################

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_deployment_path = config["prod_deployment_path"]


############################
# Read deployed ingested files
############################

with open(
    os.path.join(
        prod_deployment_path,
        "ingestedfiles.txt"
    ),
    "r"
) as f:

    ingested_files = f.read().splitlines()


############################
# Check for new data
############################

source_files = [
    file
    for file in os.listdir(input_folder_path)
    if file.endswith(".csv")
]

new_data = False

for file in source_files:

    if file not in ingested_files:
        new_data = True
        break


############################
# Stop if no new data
############################

if not new_data:

    print("No new data found.")
    quit()


print("New data found.")


############################
# Ingest newest data
############################

ingestion.merge_multiple_dataframe()


############################
# Score deployed model
############################

new_score = scoring.score_model()

with open(
    os.path.join(
        prod_deployment_path,
        "latestscore.txt"
    ),
    "r"
) as f:

    deployed_score = float(f.read())


############################
# Check for drift
############################

if new_score >= deployed_score:

    print("No model drift detected.")
    quit()


print("Model drift detected.")


############################
# Retrain model
############################

training.train_model()


############################
# Generate new score
############################

new_score = scoring.score_model()


############################
# Redeploy
############################

deployment.store_model_into_pickle()


############################
# Reporting
############################

reporting.score_model()

if os.path.exists("confusionmatrix.png"):

    shutil.copy(
        "confusionmatrix.png",
        "confusionmatrix2.png"
    )


############################
# API calls
############################

subprocess.run(
    ["python", "apicalls.py"]
)

if os.path.exists("apireturns.txt"):

    shutil.copy(
        "apireturns.txt",
        "apireturns2.txt"
    )

print("Model redeployed successfully.")