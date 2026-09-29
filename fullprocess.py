import os
import json

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting


################## Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_deployment_path = config["prod_deployment_path"]


################## Check and read new data

# Read previously ingested files
with open(
    os.path.join(
        prod_deployment_path,
        "ingestedfiles.txt"
    ),
    "r"
) as file:

    ingested_files = file.read().splitlines()

# Current source files
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


################## Deciding whether to proceed, part 1

if not new_data:
    print("No new data found")
    quit()


################## Run ingestion and training

ingestion.merge_multiple_dataframe()

training.train_model()

new_score = scoring.score_model()


################## Checking for model drift

with open(
    os.path.join(
        prod_deployment_path,
        "latestscore.txt"
    ),
    "r"
) as file:

    deployed_score = float(file.read())

model_drift = new_score < deployed_score


################## Deciding whether to proceed, part 2

if not model_drift:
    print("No model drift detected")
    quit()


################## Re-deployment

deployment.store_model_into_pickle()


################## Diagnostics and reporting

diagnostics.model_predictions()

diagnostics.dataframe_summary()

diagnostics.execution_time()

diagnostics.outdated_packages_list()

reporting.score_model()

print("Model drift detected")
print("Model redeployed successfully")