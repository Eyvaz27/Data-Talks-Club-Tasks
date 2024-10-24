import flask
import pickle 
import numpy as np
from flask import Flask, jsonify, request

default_customer = {"job": "management", "duration": 400, "poutcome": "success"}

dict_transformer_path = "/workspaces/Data-Talks-Club-Tasks/Machine Learning Zoomcamp/week5/homework_solution/homework/dv.bin"
logR_model_path = "/workspaces/Data-Talks-Club-Tasks/Machine Learning Zoomcamp/week5/homework_solution/homework/model1.bin"

with open(dict_transformer_path, 'rb') as inF:
    dict_transformer = pickle.load(inF)

with open(logR_model_path, "rb") as inF:
    logR_model = pickle.load(inF)

def predict_proba(customer):
    customer_dict = dict_transformer.transform([customer])
    subSc_pred = np.round(logR_model.predict_proba(customer_dict)[0, 1], decimals=3)
    return subSc_pred, bool(subSc_pred > 0.5)


app = Flask("Subscription")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if flask.request.method == "POST":

        customer = request.get_json()
        probability, decision = predict_proba(customer)
        result = {"Probability": probability, "Decision": decision, "is_valid": True}
        
        return jsonify(result)
    
    elif flask.request.method == "GET":
        print("Warning!!! You are obtaining results for random customer")

        customer = default_customer
        probability, decision = predict_proba(customer)
        result = {"Probability": probability, "Decision": decision, "is_valid": False}
        
        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)