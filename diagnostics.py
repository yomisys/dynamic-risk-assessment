import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


################## Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = config["output_folder_path"]
test_data_path = config["test_data_path"]
prod_deployment_path = config["prod_deployment_path"]


################## Function to get model predictions
def model_predictions():
    """
    Load the deployed model and test dataset,
    then return predictions as a list.
    """

    model_file = os.path.join(
        prod_deployment_path,
        "trainedmodel.pkl"
    )

    test_file = os.path.join(
        test_data_path,
        "testdata.csv"
    )

    with open(model_file, "rb") as file:
        model = pickle.load(file)

    test_data = pd.read_csv(test_file)

    X_test = test_data[
        [
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees"
        ]
    ]

    predictions = model.predict(X_test)

    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():
    """
    Calculate the mean, median and standard deviation
    of each numerical column in finaldata.csv.

    Also calculate the percentage of missing values
    for every column.
    """

    data_file = os.path.join(
        dataset_csv_path,
        "finaldata.csv"
    )

    dataframe = pd.read_csv(data_file)

    numerical_data = dataframe.select_dtypes(
        include=[np.number]
    )

    summary_statistics = []

    for column in numerical_data.columns:
        column_statistics = [
            float(numerical_data[column].mean()),
            float(numerical_data[column].median()),
            float(numerical_data[column].std())
        ]

        summary_statistics.append(column_statistics)

    missing_percentages = (
        dataframe.isna().mean() * 100
    ).tolist()

    summary_statistics.append(missing_percentages)

    return summary_statistics


################## Function to get timings
def execution_time():
    """
    Calculate execution times for ingestion.py
    and training.py.
    """

    ingestion_time = timeit.timeit(
        stmt="subprocess.run("
             "['python', 'ingestion.py'], "
             "check=True, "
             "stdout=subprocess.DEVNULL, "
             "stderr=subprocess.DEVNULL"
             ")",
        setup="import subprocess",
        number=1
    )

    training_time = timeit.timeit(
        stmt="subprocess.run("
             "['python', 'training.py'], "
             "check=True, "
             "stdout=subprocess.DEVNULL, "
             "stderr=subprocess.DEVNULL"
             ")",
        setup="import subprocess",
        number=1
    )

    return [ingestion_time, training_time]


################## Function to check dependencies
def outdated_packages_list():
    """
    Return a table showing installed and latest
    versions of outdated Python packages.
    """

    result = subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "list",
            "--outdated",
            "--format=json"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    outdated_packages = json.loads(result.stdout)

    package_table = pd.DataFrame(outdated_packages)

    if package_table.empty:
        return pd.DataFrame(
            columns=[
                "name",
                "version",
                "latest_version",
                "latest_filetype"
            ]
        )

    return package_table


if __name__ == "__main__":
    print("Model predictions:")
    print(model_predictions())

    print("\nDataframe summary:")
    print(dataframe_summary())

    print("\nExecution times:")
    print(execution_time())

    print("\nOutdated packages:")
    print(outdated_packages_list())