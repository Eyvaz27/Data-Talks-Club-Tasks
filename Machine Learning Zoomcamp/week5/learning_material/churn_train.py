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


dict_transformer = DictVectorizer(sparse=False)
dict_transformer.fit(X_full_train.to_dict(orient="records"))

x_full_train = dict_transformer.transform(X_full_train.to_dict(orient="records"))
x_test = dict_transformer.transform(X_test.to_dict(orient="records"))

model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
model.fit(x_full_train, y_full_train)

test_score = roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
print(f"Model's generalization performance --> AUC = {np.round(test_score, decimals=3)}")

# # # let's use pickle to save the model s.t. we can later access it
output_file = f'model_{str(C).replace(".", "_")}.bin'

with open(output_file, "wb") as outF:
    pickle.dump((dict_transformer, model), outF)

print(f"Saved the model file as {output_file}")