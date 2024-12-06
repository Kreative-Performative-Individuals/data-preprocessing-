import sys
sys.path.append('C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-')
from dataprocessing_functions import machine, kpi, store_path
import pickle

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
