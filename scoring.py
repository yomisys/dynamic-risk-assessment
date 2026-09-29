from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json

#################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model():

    # load model
    with open(
        os.path.join(model_path, 'trainedmodel.pkl'),
        'rb'
    ) as file:
        model = pickle.load(file)

    # load test data
    test_data = pd.read_csv(
        os.path.join(test_data_path, 'testdata.csv')
    )

    X_test = test_data[
        [
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees'
        ]
    ]

    y_test = test_data['exited']

    # predictions
    predictions = model.predict(X_test)

    # calculate F1 score
    f1 = metrics.f1_score(y_test, predictions)

    # write score to file
    with open(
        os.path.join(model_path, 'latestscore.txt'),
        'w'
    ) as file:
        file.write(str(f1))

    return f1


if __name__ == '__main__':
    score_model()