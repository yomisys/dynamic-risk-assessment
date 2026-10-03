from flask import Flask, jsonify, request
import json
import os

import diagnostics

###########################
# Flask setup
###########################

app = Flask(__name__)

app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = config["prod_deployment_path"]

###########################
# Prediction Endpoint
###########################

@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()

    filepath = data["filepath"]

    predictions = diagnostics.model_predictions(filepath)

    return jsonify(predictions), 200


###########################
# Scoring Endpoint
###########################

@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring_endpoint():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    score_file = os.path.join(
        prod_deployment_path,
        "latestscore.txt"
    )

    with open(score_file, "r") as file:

        score = float(
            file.read().strip()
        )

    return jsonify(score), 200


###########################
# Summary Statistics Endpoint
###########################

@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats_endpoint():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    statistics = diagnostics.dataframe_summary()

    return jsonify(statistics), 200


###########################
# Diagnostics Endpoint
###########################

@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics_endpoint():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    execution_times = diagnostics.execution_time()

    missing_data = diagnostics.missing_data()

    outdated_packages = diagnostics.outdated_packages_list()

    if hasattr(outdated_packages, "to_dict"):

        outdated_packages = (
            outdated_packages
            .to_dict(orient="records")
        )

    diagnostic_results = {
        "execution_time": execution_times,
        "missing_data": missing_data,
        "outdated_packages": outdated_packages
    }

    return jsonify(
        diagnostic_results
    ), 200


###########################
# Run App
###########################

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
        threaded=True
    )
    