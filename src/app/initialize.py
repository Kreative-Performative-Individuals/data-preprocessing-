from dataprocessing_functions import machines, kpi, features, fields, update_model_ad, tdnn_forecasting_training, \
    update_model_forecast
from dataprocessing_functions import ad_exp_train, update_model_ad_exp
import config as config
import pickle
import json
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings

warnings.filterwarnings("ignore")

# Dictionary used to mantain a local version of the original kpis in order to handle the specific batches useful for
# the imputation and the data drift analysis.
info = {}

print("Here")

for m in list(machines.keys()):
    info_asset = {}  # Reset for each machine
    for id in [machines[m]]:
        info_kpi = {}  # Reset for each asset
        for k in list(kpi.keys()):
            info_op = {}
            for o in kpi[k][1]:
                info_op[o] = [[[], [], [], [], []], 0, [],
                              []]  # [0]: batch ([0][0]: 'sum', [0][1]: 'avg', [0][2]: 'min', [0][3]: 'max', [0][4] 'var'
                # [1]: counter for missing values (same subdivision of the batch)
                # [2]: trained model for anomaly detection.
                # [3]: trained model for forecasting
            info_kpi[k] = info_op
        info_asset[id] = info_kpi  # Associate KPIs with the asset ID
    info[m] = info_asset  # Associate assets with the machine type

# Save the dictionary to a pickle file
with open(config.STORE_PKL, "wb") as file:  # "wb" means write binary
    pickle.dump(info, file)

''''For the initialization we will consider historical data the first 200 days (up to 16/09) and the remaining 33 days of each timeseries (specific combination of
['asset_id', 'name', 'kpi', 'operation']) as available for the stream.
'''

with open(config.CLEANED_PREDICTED_DATA_PATH, "r") as file:
    historical = json.load(file)

historical = pd.DataFrame(historical)[:38400]
print(historical.shape)

with open(config.HISTORICAL_DATA_PATH, "w") as file:
    json.dump(historical.to_dict(), file, indent=1)

# ''''Fill the batches of each feature for each kpi and machine. 
# Train the anomaly detector for each machine and kpi over the 200 days considered for historical data ''''
for m in list(machines.keys()):
    a = machines[m]
    for k in list(kpi.keys()):
        for o in kpi[k][1]:
            print(f'combination: {m} - {a} - {k} - {o}')
            if k=='emission_factor':
                continue
                    section = historical[
                        (historical['name'] == m) & (historical['asset_id'] == a) & (historical['kpi'] == k) & (
                                historical['operation'] == o) & (historical['status'] != 'Corrupted')]

                    with open(config.STORE_PKL, "rb") as file:
                        info = pickle.load(file)
            #here there's an error        for f in features:
                        info[m][a][k][o][0][features.index(f)] = section[f].iloc[-40:].to_list()
                    with open(config.STORE_PKL, "wb") as file:
                        pickle.dump(info, file)

                    nan_columns = section.columns[section.isna().all()]
                    length = 30
                    train_set = section[features].iloc[-length:].reset_index(drop=True)
                    train_set = train_set.drop(columns=nan_columns)
                    train_set = train_set.fillna(0)

                    s = []
                    cc = np.arange(0.01, 0.5, 0.01)
                    for c in cc:
                        model = IsolationForest(n_estimators=200, contamination=c)
                        an_pred = model.fit_predict(train_set)
                        if len(set(an_pred)) > 1:  # Check for multiple clusters
                            s.append(silhouette_score(train_set, an_pred))
                        else:
                            s.append(-1)  # Append a placeholder or ignore this case
                    if max(s) <= 0.75:
                        optimal_c = 1e-5
                    else:
                        optimal_c = cc[np.argmax(s)]
                    model = IsolationForest(n_estimators=200, contamination=optimal_c)
                    explainer = LimeTabularExplainer(
                train_set.values,
                mode='classification',
                feature_names=features,
                class_names=['Normal', 'Anomaly'])

                    update_model_ad(section.iloc[0].to_dict(), model)
                    update_model_ad_exp(section.iloc[0].to_dict(), explainer)

                    predictions = model.fit_predict(train_set)
                    predictions = np.vstack([train_set.index, predictions])
                    marker_indices = predictions[0, predictions[1] == -1]  # First row corresponding to -1 in second row

            model_forecasting={}
            for f in features:
                if all(pd.isna(section[f])):
                    continue
                else:
                    train_set=section[['time'] + [f]]
                    train_set=train_set.fillna(0)
                    print(train_set)
                    model=tdnn_forecasting_training(train_set)
                    model_forecasting[f]=model
                    print(model)
            print(model_forecasting)
            update_model_forecast(section.iloc[0].to_dict(), model_forecasting)
