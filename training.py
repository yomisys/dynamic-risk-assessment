from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for training the model
def train_model():

    # read ingested data
    data = pd.read_csv(
        os.path.join(dataset_csv_path, 'finaldata.csv')
    )

    # define features and target
    X = data[
        [
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees'
        ]
    ]

    y = data['exited']

    # create logistic regression model
    model = LogisticRegression(
        random_state=0,
        solver='liblinear'
    )

    # train model
    model.fit(X, y)

    # ensure output folder exists
    os.makedirs(model_path, exist_ok=True)

    # save model
    with open(
        os.path.join(model_path, 'trainedmodel.pkl'),
        'wb'
    ) as file:

        pickle.dump(model, file)


if __name__ == '__main__':
    train_model()

