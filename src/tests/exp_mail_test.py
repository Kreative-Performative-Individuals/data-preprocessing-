import sys
import os

import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.app import config
from src.app.dataprocessing_functions import ad_train, cleaning_pipeline, ad_exp_train, ad_exp_predict, ad_predict
from src.app.connection_functions import get_datapoint, get_historical_data_mock
from src.app.connection_functions import send_alert
import pandas as pd
from sklearn.ensemble import IsolationForest
from lime.lime_tabular import LimeTabularExplainer
import pickle
import json
import time

c = 0
new_datapoint = get_datapoint(c)  # CONNECTION WITH API
cleaned_datapoint = cleaning_pipeline(new_datapoint)# use this to decide which model to train

# print('Training a model for the following datapoint:')
# print(json.dumps(cleaned_datapoint, indent=4))
historical_data = get_historical_data_mock(cleaned_datapoint['name'], cleaned_datapoint['asset_id'],
                                                      cleaned_datapoint['kpi'], cleaned_datapoint['operation'], "2024-02-17 00:00:00+00:00",
                                                      cleaned_datapoint['time'])  # CONNECTION WITH API

train_set = pd.DataFrame(historical_data)
nan_columns = train_set.columns[train_set.isna().all()]
train_set = train_set.drop(columns=nan_columns)
train_set = train_set.drop(columns=['time', 'asset_id', 'name', 'kpi', 'operation', 'status'])
train_set = train_set.fillna(0)

#train_set = train_set * 1000000
#print(train_set)

model = IsolationForest(n_estimators=200, contamination='auto')
model.fit(train_set.values)
explainer = ad_exp_train(train_set)


point = train_set.iloc[0]
point['sum'] = 293184329187
point['avg'] = 192834981234
point['min'] = 984037098712340978312

pred = model.predict([point])

class_pred = lambda x: [0.01, 0.99] if model.predict([x])[0] == 1 else [0.99, 0.01]
print(f"Prediction: {pred}")
print(f"Prediction: {class_pred(point)}")
predd_legenda = ['Anomaly', 'Normal']
prediction = np.argmax(class_pred(point))
print(f"Prediction: {predd_legenda[prediction]}")
prediction = predd_legenda[prediction]

if prediction == 'Anomaly':
    point_dict = point.to_dict()
    out = ad_exp_predict(point_dict,  explainer=explainer, model=model)
    cleaned_datapoint['status'] = 'anomaly'
    cleaned_datapoint['explanation'] = out
    anomaly_score = model.decision_function([point])
    anomaly_prob = 1 - (1 / (1 + np.exp(-5 * anomaly_score)))
    anomaly_prob = int(anomaly_prob[0] * 100)
    print(f"Anomaly probability: {anomaly_prob}%")
    send_alert(cleaned_datapoint, 'Anomaly', probability=anomaly_prob)
    
