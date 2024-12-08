from src.app.dataprocessing_functions import machine, kpi, store_path
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
            info_op = {}
            for o in kpi[k][1]:
                info_op[o] = [
                    [[], [], [], [], []],
                    0,
                    [],
                ]  # [0]: batch ([0][0]: 'sum', [0][1]: 'avg', [0][2]: 'min', [0][3]: 'max', [0][4] 'var'
                # [1]: counter for missing values (same subdivision of the batch)
                # [2]: trained model for anomaly detection.
            info_kpi[k] = info_op
        info_asset[id] = info_kpi  # Associate KPIs with the asset ID
    info[m] = info_asset  # Associate assets with the machine type


# Save the dictionary to a pickle file
with open(store_path, "wb") as file:  # "wb" means write binary
    pickle.dump(info, file)


"""'For the initialization we will consider historical data the first 200 days (up to 16/09) and the remaining 33 days of each timeseries (specific combination of
['asset_id', 'name', 'kpi', 'operation']) as available for the stream.
"""


# Define a relative path
relative_path = os.path.join("initialization", "cleaned_predicted_dataset.json")
script_dir = os.getcwd()
absolute_path = os.path.join(script_dir, relative_path)

with open(absolute_path, "r") as file:
    historical = json.load(file)

historical = pd.DataFrame(historical)[:38400]

# Define a relative path
relative_path = os.path.join("initialization", "original_adapted_dataset.json")
script_dir = os.getcwd()
absolute_path = os.path.join(script_dir, relative_path)

with open(absolute_path, "r") as file:
    stream = json.load(file)

stream = pd.DataFrame(stream)[38400:]
