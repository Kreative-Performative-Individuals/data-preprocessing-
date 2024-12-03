from dataprocessing_functions import machine, kpi, store_path, discarded_path
import json

    # Dictionary used to mantain a local version of the original kpis in order to handle the specific batches useful for the imputation and the data drift analysis.
info = {}

for m in list(machine.keys()):
    info_asset = {}  # Reset for each machine
    for id in list(machine[m]):
        info_kpi = {}  # Reset for each asset
        for k in list(kpi.keys()):
            info_op={}
            for o in kpi[k][1]:
                info_op[o] = [[[], [], [], [], []], [], [], []]  # [0]: batch ([0][0]: 'sum', [0][1]: 'avg', [0][2]: 'min', [0][3]: 'max', [0][4] 'var'
                                                                        # [1]: counter for missing values (same subdivision of the batch)
                                                                        # [2]: trained model for anomaly detection.
            info_kpi[k] = info_op
        info_asset[id] = info_kpi  # Associate KPIs with the asset ID
    info[m] = info_asset  # Associate assets with the machine type


# Save the dictionary to a JSON file
with open(store_path, "w") as json_file:
    json.dump(info, json_file, indent=1) 

with open(discarded_path, "w") as json_file:
    discarded_dp=[]
    json.dump(discarded_dp, json_file, indent=1) 