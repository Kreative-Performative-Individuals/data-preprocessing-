import sys
import os


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

print('Training a model for the following datapoint:')
print(json.dumps(cleaned_datapoint, indent=4))
historical_data = get_historical_data_mock(cleaned_datapoint['name'], cleaned_datapoint['asset_id'],
                                                      cleaned_datapoint['kpi'], cleaned_datapoint['operation'], "2024-02-17 00:00:00+00:00",
                                                      cleaned_datapoint['time'])  # CONNECTION WITH API

train_set = pd.DataFrame(historical_data)
nan_columns = train_set.columns[train_set.isna().all()]
train_set = train_set.drop(columns=nan_columns)
train_set = train_set.drop(columns=['time', 'asset_id', 'name', 'kpi', 'operation', 'status'])
train_set = train_set.fillna(0)

print(train_set.head())
point = train_set.iloc[0]
print(f"Point: {point}")
#point = point.to_dict()
#print(f"Point: {point}")
#point['var'] = 0


time_start = time.time()
model = IsolationForest(n_estimators=200, contamination='auto')
model.fit(train_set)
explainer = LimeTabularExplainer(
    train_set.values,
    mode='classification',
    feature_names=[col for col in train_set.columns],
    class_names=['Normal', 'Anomaly']
    )
end_time = time.time()
print(f"Training time: {end_time - time_start}")

pred = model.predict([point])
print(f"Prediction: {pred}")
exp = explainer.explain_instance(point, 
                                 model.predict, 
                                 labels=[0, 1],
                                 num_features=5,
                                 )
print(f"Explanation: {exp}")
# with open('data/store.pkl', 'rb') as file:
#     data = pickle.load(file)
#     print(json.dumps(data, indent=4))