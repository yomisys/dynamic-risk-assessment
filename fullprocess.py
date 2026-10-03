import os
import json
import shutil
import subprocess

import ingestion
import training
import scoring
import deployment


############################
# Load configuration
############################

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
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

    deployed_files = f.read().splitlines()


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

    if file not in deployed_files:

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
# Ingest new data
############################

ingestion.merge_multiple_dataframe()


############################
# Train candidate model
############################

training.train_model()

candidate_f1 = scoring.score_model()


############################
# Read deployed score
############################

with open(
    os.path.join(
        prod_deployment_path,
        "latestscore.txt"
    ),
    "r"
) as f:

    deployed_f1 = float(f.read())


############################
# Compare candidate vs deployed
############################

print(
    f"Candidate F1: {candidate_f1:.3f} | "
    f"Deployed F1: {deployed_f1:.3f}"
)

if candidate_f1 <= deployed_f1:

    print(
        "Candidate is not better than "
        "the deployed model. "
        "Keeping the current deployment."
    )

    quit()


############################
# Deploy better model
############################

print("Better model found.")
print("Deploying new model.")

deployment.store_model_into_pickle()


############################
# Run reporting
############################

subprocess.run(
    ["python", "reporting.py"]
)

if os.path.exists("confusionmatrix.png"):

    shutil.copy(
        "confusionmatrix.png",
        "confusionmatrix2.png"
    )


############################
# Run API calls
############################

subprocess.run(
    ["python", "apicalls.py"]
)

if os.path.exists("apireturns.txt"):

    shutil.copy(
        "apireturns.txt",
        "apireturns2.txt"
    )

print("Process complete.")