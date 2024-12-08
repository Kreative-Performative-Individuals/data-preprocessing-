from dataprocessing_functions import machine, kpi, store_path
import pickle
import json
import os
import pandas as pd
    # Dictionary used to mantain a local version of the original kpis in order to handle the specific batches useful for the imputation and the data drift analysis.
info = {}

for m in list(machine.keys()):
    info_asset = {}  # Reset for each machine
    for id in list(machine[m]):
        info_kpi = {}  # Reset for each asset
        for k in list(kpi.keys()):
            info_op={}
            for o in kpi[k][1]:
                info_op[o] = [[[], [], [], [], []], 0, []]  # [0]: batch ([0][0]: 'sum', [0][1]: 'avg', [0][2]: 'min', [0][3]: 'max', [0][4] 'var'
                                                                        # [1]: counter for missing values (same subdivision of the batch)
                                                                        # [2]: trained model for anomaly detection.
            info_kpi[k] = info_op
        info_asset[id] = info_kpi  # Associate KPIs with the asset ID
    info[m] = info_asset  # Associate assets with the machine type


# Save the dictionary to a pickle file
with open(store_path, "wb") as file:  # "wb" means write binary
    pickle.dump(info, file)


''''For the initialization we will consider historical data the first 200 days (up to 16/09) and the remaining 33 days of each timeseries (specific combination of
['asset_id', 'name', 'kpi', 'operation']) as available for the stream.
'''

with open(os.path.join(os.getcwd(), os.path.join("initialization", "cleaned_predicted_dataset.json")), "r") as file:
    historical=json.load(file)

historical=pd.DataFrame(historical)[:38400]

with open(os.path.join(os.getcwd(), os.path.join("initialization", "historical_dataset.json")), "w") as file:
    json.dump(historical.to_dict(), file, indent=1)

# # ''''Fill the batches of each feature for each kpi and machine. 
# # Train the anomaly detector for each machine and kpi over the 200 days considered for historical data ''''

# from dataprocessing_functions import features, fields, update_model_ad
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import silhouette_score
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
# for m in list(machine.keys()):
#     for a in machine[m]:
#         for k in list(kpi.keys())[0]:
#             try:
#                 for o in kpi[k][1][0]:
#                     section=historical[(historical['name']==m)&(historical['asset_id']==a)&(historical['kpi']==k)&(historical['operation']==o)&(historical['status']!='Corrupted')]
#                     with open(store_path, "rb") as file:
#                         info = pickle.load(file)
#                     for f in features:
#                         info[m][a][k][o][0][features.index(f)]=section[f].iloc[-40:].to_list()
#                     nan_columns = section.columns[section.isna().all()]
#                     with open(store_path, "wb") as file:
#                         pickle.dump(info, file)
#                     length=30
#                     train_set=section[features].iloc[-length:].reset_index(drop=True)
#                     train_set = train_set.drop(columns=nan_columns)
#                     train_set=train_set.fillna(0)

#                     s=[]
#                     cc=np.arange(0.01, 0.5, 0.01)
#                     for c in cc:
#                         model = IsolationForest(n_estimators=200, contamination=c)
#                         an_pred=model.fit_predict(train_set)
#                         if len(set(an_pred)) > 1:  # Check for multiple clusters
#                             s.append(silhouette_score(train_set, an_pred))
#                         else:
#                             s.append(-1)  # Append a placeholder or ignore this case
#                     if max(s)<=0.75:
#                         optimal_c=1e-5
#                     else:
#                         optimal_c=cc[np.argmax(s)]
#                     model = IsolationForest(n_estimators=200, contamination=optimal_c)
#                     update_model_ad(section.iloc[0].to_dict(), model)
#                     predictions = model.fit_predict(train_set)
#                     predictions= np.vstack([train_set.index, predictions])

#                     marker_indices = predictions[0, predictions[1] == -1]  # First row corresponding to -1 in second row
#                     plt.figure(figsize=(10, 6)) 
#                     for f in train_set.columns:
#                         plt.plot(train_set.index, train_set[f], label=f)
#                         plt.scatter(marker_indices, train_set[f].iloc[marker_indices], color="red", label="Anomalies")
                        
#                     plt.title('Train set')
#                     plt.xlabel('Time')
#                     plt.ylabel('Value')
#                     plt.legend()
#                     plt.tight_layout()
#                     output_dir = "anomaly_training"
#                     os.makedirs(output_dir, exist_ok=True)
#                     save_path = os.path.join(output_dir, f"{m}_{a}_{k}_{o}.png")
#                     plt.savefig(save_path)
#                     plt.close()
#             except KeyError:
#                 continue


