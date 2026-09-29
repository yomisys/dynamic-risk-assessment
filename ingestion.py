import pandas as pd
import os
import json

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def merge_multiple_dataframe():

    final_dataframe = pd.DataFrame()
    ingested_files = []

    for file in os.listdir(input_folder_path):

        if file.endswith(".csv"):

            filepath = os.path.join(input_folder_path, file)

            temp_df = pd.read_csv(filepath)

            final_dataframe = pd.concat(
                [final_dataframe, temp_df],
                ignore_index=True
            )

            ingested_files.append(file)

    final_dataframe.drop_duplicates(inplace=True)

    os.makedirs(output_folder_path, exist_ok=True)

    final_dataframe.to_csv(
        os.path.join(output_folder_path, "finaldata.csv"),
        index=False
    )

    with open(
        os.path.join(output_folder_path, "ingestedfiles.txt"),
        "w"
    ) as f:

        for file in ingested_files:
            f.write(file + "\n")


if __name__ == "__main__":
    merge_multiple_dataframe()
    