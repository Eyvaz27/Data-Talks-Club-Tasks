import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

C = 1.0

raw_data = pd.read_csv("/app/new/splatter-image/DTC/Machine Learning Zoomcamp/week3/bank-full.csv", sep=";")
raw_data.columns = raw_data.columns.str.lower().str.replace(' ', '_')

used_columns = ["age", "job", "marital", "education", "balance", 
                "housing", "contact", "day", "month", "duration", 
                "campaign", "pdays", "previous", "poutcome", "y"]
sample_data = raw_data.loc[:, used_columns]


categorical_columns = sample_data.columns[sample_data.dtypes == "object"].values[:-1]
numerical_columns = sample_data.columns[sample_data.dtypes != "object"].values

X = sample_data.iloc[:, :-1]
y = sample_data["y"].map({'yes': 1.0, 'no': 0.0})

X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_full_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

y_full_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


def predict(X, dict_transformer, model):
    x_dict = dict_transformer.transform(X.to_dict(orient="records"))
    return model.predict_proba(x_dict)[:, 1]

output_file = f'model_{str(C).replace(".", "_")}.bin'

with open(output_file, "rb") as inF:
    dict_transformer, model = pickle.load(inF)

print(f"Loaded the model and dictionary transformer from {output_file}")

test_score = roc_auc_score(y_true=y_test, y_score=predict(X_test, dict_transformer, model))
print(f"Model's generalization performance --> AUC = {np.round(test_score, decimals=3)}")

# # # let's evaluate model on new customer
customer = {
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

x_customer = dict_transformer.transform([customer])
churn_prediction = np.round(model.predict_proba(x_customer)[0, 1], decimals=3)
print(f"New customer will churn with probability {churn_prediction}")

# # # run the files with  -> python -m file_name