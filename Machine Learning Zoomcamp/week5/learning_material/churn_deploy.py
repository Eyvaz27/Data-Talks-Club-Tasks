import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import flask
from flask import Flask
from flask import request
from flask import jsonify
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score


default_customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

C = 1.0

output_file = f'model_{str(C).replace(".", "_")}.bin'

with open(output_file, "rb") as inF:
    dict_transformer, model = pickle.load(inF)

print(f"Loaded the model and dictionary transformer from {output_file}")

def predict_proba(customer):
    x_customer = dict_transformer.transform([customer])
    churn_prediction = np.round(model.predict_proba(x_customer)[0, 1], decimals=3)
    return churn_prediction, bool(churn_prediction>0.5)

app = Flask("churn")
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if flask.request.method == 'POST':
        customer = request.get_json()
        churn_prediction, decision = predict_proba(customer)
        
        result = {"churn_probability": churn_prediction, "decision": decision, "is_valid": True}
        return jsonify(result)
    if flask.request.method == 'GET':
        print("Warning!! You are getting results for random customer")
        customer = default_customer
        churn_prediction, decision = predict_proba(customer)
        
        result = {"churn_probability": churn_prediction, "decision": decision, "is_valid": False}
        return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

# # # run the files with  -> python -m file_name