import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##############Function for reporting
def score_model():

    # load deployed model
    with open(
        os.path.join(prod_deployment_path, 'trainedmodel.pkl'),
        'rb'
    ) as file:
        model = pickle.load(file)

    # load test data
    test_data = pd.read_csv(
        os.path.join(dataset_csv_path, 'testdata.csv')
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
    y_pred = model.predict(X_test)

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.tight_layout()

    plt.savefig('confusionmatrix.png')

    plt.close()


if __name__ == '__main__':
    score_model()
    