import pandas as pd
import numpy as np
import pickle
import timeit
import subprocess
import json
import os


################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


################## Function to get model predictions
def model_predictions(test_data_file=None):

    if test_data_file is None:
        test_data_file = os.path.join(
            test_data_path,
            'testdata.csv'
        )

    with open(
        os.path.join(
            prod_deployment_path,
            'trainedmodel.pkl'
        ),
        'rb'
    ) as file:

        model = pickle.load(file)

    test_data = pd.read_csv(test_data_file)

    X = test_data[
        [
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees'
        ]
    ]

    predictions = model.predict(X)

    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():

    data = pd.read_csv(
        os.path.join(
            dataset_csv_path,
            'finaldata.csv'
        )
    )

    numerical_columns = [
        'lastmonth_activity',
        'lastyear_activity',
        'number_of_employees',
        'exited'
    ]

    # Mean
    means = data[numerical_columns].mean().tolist()

    # Median
    medians = data[numerical_columns].median().tolist()

    # Mode
    modes = data[numerical_columns].mode().iloc[0].tolist()

    # Standard Deviation
    stds = data[numerical_columns].std().tolist()

    return [
        means,
        medians,
        modes,
        stds
    ]


################## Function to measure missing data
def missing_data():

    data = pd.read_csv(
        os.path.join(
            dataset_csv_path,
            'finaldata.csv'
        )
    )

    missing_percentages = []

    for column in data.columns:

        percentage = (
            data[column]
            .isnull()
            .sum()
            / len(data)
        ) * 100

        missing_percentages.append(percentage)

    return missing_percentages


################## Function to get timings
def execution_time():

    start_time = timeit.default_timer()

    os.system('python ingestion.py')

    ingestion_time = (
        timeit.default_timer()
        - start_time
    )

    start_time = timeit.default_timer()

    os.system('python training.py')

    training_time = (
        timeit.default_timer()
        - start_time
    )

    return [
        ingestion_time,
        training_time
    ]


################## Function to check dependencies
def outdated_packages_list():

    outdated = subprocess.check_output(
        [
            'pip',
            'list',
            '--outdated'
        ]
    ).decode('utf-8')

    return outdated


if __name__ == '__main__':

    print("Predictions:")
    print(model_predictions())

    print("\nSummary Statistics:")
    print(dataframe_summary())

    print("\nMissing Data:")
    print(missing_data())

    print("\nExecution Times:")
    print(execution_time())

    print("\nOutdated Packages:")
    print(outdated_packages_list())
    