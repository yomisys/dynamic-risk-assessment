import pandas as pd
import numpy as np
import pickle
import timeit
import subprocess
import json
import os
import sys
from importlib.metadata import version, PackageNotFoundError


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

    means = data[numerical_columns].mean().tolist()

    medians = data[numerical_columns].median().tolist()

    modes = data[numerical_columns].mode().iloc[0].tolist()

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

    rows = []

    with open('requirements.txt', 'r') as f:

        modules = []

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith('#'):
                continue

            package = (
                line.split('==')[0]
                .split('>=')[0]
                .split('<=')[0]
                .strip()
            )

            modules.append(package)

    for module in modules:

        try:
            installed = version(module)

        except PackageNotFoundError:
            installed = 'not installed'

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'index',
                    'versions',
                    module
                ],
                capture_output=True,
                text=True
            ).stdout

            if '(' in result and ')' in result:

                latest = (
                    result
                    .split('(')[1]
                    .split(')')[0]
                )

            else:

                latest = 'unknown'

        except Exception:

            latest = 'unknown'

        rows.append(
            {
                'module': module,
                'installed': installed,
                'latest': latest
            }
        )

    return pd.DataFrame(rows)


if __name__ == '__main__':

    print("Predictions:")
    print(model_predictions())

    print("\nSummary Statistics:")
    print(dataframe_summary())

    print("\nMissing Data:")
    print(missing_data())

    print("\nExecution Times:")
    print(execution_time())

    print("\nDependency Check:")
    print(outdated_packages_list())
    