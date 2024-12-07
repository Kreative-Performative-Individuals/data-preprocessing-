
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys
sys.path.append('C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-')
from dataprocessing_functions import features, data_path
import json

#This is the identity of the stream to generate.
id={'asset_id':  'ast-yhccl1zjue2t',
    'name': 'metal_cutting',
    'kpi': 'time',
    'operation': 'working'}

# np.random.seed(42)
data=[]
anomaly_probability=np.random.uniform(0.01, 0.035)
probability_of_nan = 0.05
#_____________________________________________________________________________________________________________________________________________________________________________________
# historical data

# Start date
n_points_historical = 1000



normal_data=[]
anomaly_indices = np.random.choice(range(n_points_historical), size=int(n_points_historical * anomaly_probability), replace=False)
magg_factor=np.random.uniform(1.2, 1.5, size=int(n_points_historical * anomaly_probability))
window_size=15

normal_data.append(np.convolve(np.random.normal(60000, 9000, n_points_historical), np.ones(window_size)/window_size, mode='same')) #sum #86400  
normal_data[0][anomaly_indices] = max(normal_data[0])*magg_factor

normal_data.append(np.convolve(np.random.normal(31100, 9000, n_points_historical),  np.ones(window_size)/window_size, mode='same')) #avg 
normal_data[1][anomaly_indices] = max(normal_data[1])*magg_factor

normal_data.append(np.convolve(np.random.normal(15000, 8000, n_points_historical),  np.ones(window_size)/window_size, mode='same')) #min 
normal_data[2][anomaly_indices] = max(normal_data[2])*magg_factor

normal_data.append(np.convolve(np.random.normal(44200, 3000, n_points_historical),  np.ones(window_size)/window_size, mode='same')) #max 
normal_data[3][anomaly_indices] = max(normal_data[3])*magg_factor

normal_data.append(np.convolve(np.random.normal(35000, 8000, n_points_historical),  np.ones(window_size)/window_size, mode='same')) #var
normal_data[4][anomaly_indices] = max(normal_data[4])*magg_factor

anomaly_status = ['Normal'] * n_points_historical
for idx in anomaly_indices:
    anomaly_status[idx] = 'Anomaly'

start_date = datetime(2023, 5, 15, 00, 00, tzinfo=timezone.utc)
times = pd.date_range(
    start=start_date, 
    periods=n_points_historical, 
    freq='D',  # Daily frequency
    tz=timezone.utc  # Set timezone
)


# Create a DataFrame with the time index
historical_data = {
    'time': times[window_size:-window_size].astype(str).tolist(),
    'asset_id':  [id['asset_id']] * len(times[window_size:-window_size]),  # Added asset_id column
    'name': [id['name']] * len(times[window_size:-window_size]),  # Added name column
    'kpi': [id['kpi']] * len(times[window_size:-window_size]),  # Added kpi column
    'operation': [id['operation']] * len(times[window_size:-window_size]),  # Added operation column
    'sum': normal_data[0][window_size:-window_size].tolist(),
    'avg': normal_data[1][window_size:-window_size].tolist(),
    'min': normal_data[2][window_size:-window_size].tolist(),
    'max': normal_data[3][window_size:-window_size].tolist(),
    'var': normal_data[4][window_size:-window_size].tolist(), 
    'status': anomaly_status[window_size:-window_size]
}


# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# for f in features:
#     plt.plot(historical_data['time'], historical_data[f], label=f)

# plt.title('Historical data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.xticks(range(1, n_points_historical, 10))
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

data.append(historical_data) 
##_____________________________________________________________________________________________________________________________________________________________________________________
## stream data

# Start date
n_points_stream=500
drift_index=np.random.uniform(100,n_points_stream-300) 
increment=13000

# Generate normal data (mean=0, std=1) with outliers
normal_data=[]
non_nan_indices=[]
magg_factor=np.random.uniform(1.2, 1.5, size=int(n_points_stream * anomaly_probability))

normal_data.append(np.convolve(np.random.normal(60000, 9000, n_points_stream), np.ones(window_size)/window_size, mode='same')) #sum #86400  
normal_data[0]=np.array([value + increment if index > drift_index else value for index, value in enumerate(normal_data[0])])
nan_mask = np.random.rand(n_points_stream) < probability_of_nan
nan_mask[0]=False
normal_data[0][nan_mask] = np.nan
non_nan_indices.append(list(np.where(~nan_mask)[0]))

normal_data.append(np.convolve(np.random.normal(31100, 9000, n_points_stream), np.ones(window_size)/window_size, mode='same')) #avg 
normal_data[1]=np.array([value + increment if index > drift_index else value for index, value in enumerate(normal_data[1])])
nan_mask = np.random.rand(n_points_stream) < probability_of_nan
nan_mask[0]=False
normal_data[1][nan_mask] = np.nan
non_nan_indices.append(list(np.where(~nan_mask)[0]))

normal_data.append(np.convolve(np.random.normal(15000, 8000, n_points_stream), np.ones(window_size)/window_size, mode='same')) #min 
normal_data[2]=np.array([value + increment if index > drift_index else value for index, value in enumerate(normal_data[2])])
nan_mask = np.random.rand(n_points_stream) < probability_of_nan
nan_mask[0]=False
normal_data[2][nan_mask] = np.nan
non_nan_indices.append(list(np.where(~nan_mask)[0]))

normal_data.append(np.convolve(np.random.normal(44200, 7000, n_points_stream), np.ones(window_size)/window_size, mode='same')) #max 
normal_data[3]=np.array([value + increment if index > drift_index else value for index, value in enumerate(normal_data[3])])
nan_mask = np.random.rand(n_points_stream) < probability_of_nan
nan_mask[0]=False
normal_data[3][nan_mask] = np.nan
non_nan_indices.append(list(np.where(~nan_mask)[0]))

normal_data.append(np.convolve(np.random.normal(35000, 8000, n_points_stream), np.ones(window_size)/window_size, mode='same')) #var
normal_data[4]=np.array([value + increment if index > drift_index else value for index, value in enumerate(normal_data[4])])
nan_mask = np.random.rand(n_points_stream) < probability_of_nan
nan_mask[0]=False
normal_data[4][nan_mask] = np.nan
non_nan_indices.append(list(np.where(~nan_mask)[0]))

possible_anomaly_indeces=list(set(non_nan_indices[0]).intersection(non_nan_indices[1], non_nan_indices[2], non_nan_indices[3], non_nan_indices[4]))
anomaly_indices = np.random.choice(possible_anomaly_indeces, size=int(n_points_stream * anomaly_probability), replace=False)
normal_data[0][anomaly_indices] = max(normal_data[0])*magg_factor
normal_data[1][anomaly_indices] = max(normal_data[1])*magg_factor
normal_data[2][anomaly_indices] = max(normal_data[2])*magg_factor
normal_data[3][anomaly_indices] = max(normal_data[3])*magg_factor
normal_data[4][anomaly_indices] = max(normal_data[4])*magg_factor

# p_o=5
# anomalies_indices = np.random.choice(n_points_stream, size=int((p_o/100)*n_points_stream), replace=False)  
# for i in range(5): 
#     normal_data[i][anomalies_indices] += np.random.normal(10, 5, int((p_o/100)*n_points_stream))  


start_date = historical_data['time'][-window_size]
times = pd.date_range(
    start=start_date, 
    periods=n_points_stream, 
    freq='D',  # Daily frequency
    tz=timezone.utc  # Set timezone
)
times=times[window_size:-window_size].astype(str).tolist()
print(f'Drift timestamp: {times[int(drift_index)]}')

# Create a DataFrame with the time index
stream_data = {
    'time': times,
    'asset_id':  [id['asset_id']] * len(times),  # Added asset_id column
    'name': [id['name']] * len(times),  # Added name column
    'kpi': [id['kpi']] * len(times),  # Added kpi column
    'operation': [id['operation']] * len(times),  # Added operation column
    'sum': normal_data[0][window_size:-window_size].tolist(),
    'avg': normal_data[1][window_size:-window_size].tolist(),
    'min': normal_data[2][window_size:-window_size].tolist(),
    'max': normal_data[3][window_size:-window_size].tolist(),
    'var': normal_data[4][window_size:-window_size].tolist(),
    'status': ['A/N']* len(times)
}

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for f in features:
    #plt.plot(historical_data['time'], historical_data[f], label=f)
    plt.plot(pd.Series(stream_data['time']).str[:10], stream_data[f], label=f)

plt.title('stream data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(range(1, n_points_stream, 10))
plt.legend()
plt.xticks(rotation=90)
plt.ylim([0,100000])
plt.tight_layout()
plt.savefig("C:\\Users\\mcapo\\Desktop\\dati sintetici.png", dpi=300)
plt.show()

data.append(stream_data)


with open(data_path, "w") as json_file:
    json.dump(data, json_file, indent=1) 

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
train_set=pd.DataFrame(historical_data)[features]
s=[]
cc=np.arange(0.01, 0.5, 0.01)
for c in cc:
    model = IsolationForest(n_estimators=200, contamination=c)
    an_pred=model.fit_predict(train_set)
    s.append(silhouette_score(train_set, an_pred))
optimal_c=cc[np.argmax(s)]
model = IsolationForest(n_estimators=200, contamination=optimal_c)
model.fit_predict(train_set)

from dataprocessing_functions import update_model_ad
update_model_ad(id, model)

##_____________________________________________________________________________________________________________________________________________________________________________________
## transform the dataset given by the professor

# import numpy as np
# import pandas as pd
# import sys
# sys.path.append('C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-')

# from dataprocessing_functions import machine

# # # Now you can use 'machine' as needed
# # print(machine)


# # machine={'metal_cutting': ['ast-yhccl1zjue2t', 'ast-ha448od5d6bd', 'ast-6votor3o4i9l', 'ast-5aggxyk5hb36', 'ast-anxkweo01vv2', 'ast-6nv7viesiao7'],
# # 'laser_cutting': ['ast-xpimckaf3dlf'],
# # 'laser_welding': ['ast-hnsa8phk2nay', 'ast-206phi0b9v6p'],
# # 'assembly': ['ast-pwpbba0ewprp', 'ast-upqd50xg79ir', 'ast-sfio4727eub0'],
# # 'testing': ['ast-nrd4vl07sffd', 'ast-pu7dfrxjf2ms', 'ast-06kbod797nnp'],
# # 'riveting': ['ast-o8xtn5xa8y87']}

# #code to assign anomaly
# file_path = 'C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-\\initialization\\smart_app_data.pkl'
# df = pd.read_pickle(file_path)

# asset_ids = df['asset_id'].unique().tolist()
# for a in asset_ids:
#     key = [key for key, val in machine.items() if a in val]
#     df.loc[df['asset_id'] == a, 'name'] = key[0]
# df['operation']=np.nan
# df.loc[df['kpi']=='working_time', 'operation']='working'
# df.loc[df['kpi']=='working_time', 'kpi']='time'
# df.loc[df['kpi']=='idle_time', 'operation']='idle'
# df.loc[df['kpi']=='idle_time', 'kpi']='time'
# df.loc[df['kpi']=='offline_time', 'operation']='offline'
# df.loc[df['kpi']=='offline_time', 'kpi']='time'
# # Corrected code to drop rows where 'kpi' is 'cost_working' or 'cost_idle'
# df.drop(df[df['kpi'] == 'cost_working'].index, inplace=True)
# df.drop(df[df['kpi'] == 'cost_idle'].index, inplace=True)
# df.loc[df['kpi']=='consumption', 'operation']='working'
# df.loc[df['kpi']=='consumption_idle', 'operation']='offline'
# df.loc[df['kpi']=='consumption_working', 'operation']='idle'
# df.loc[df['kpi']=='consumption_idle', 'kpi']='consumption'
# df.loc[df['kpi']=='consumption_working', 'kpi']='consumption'
# df.loc[df['kpi']=='power', 'operation']='independent'
# df.loc[df['kpi']=='cost', 'operation']='independent'
# df.loc[df['kpi']=='cycles', 'operation']='working'
# df.loc[df['kpi']=='good_cycles', 'operation']='working'
# df.loc[df['kpi']=='bad_cycles', 'operation']='working'
# df.loc[df['kpi']=='average_cycle_time', 'operation']='working'
# import json
# df_dict = df.reset_index(drop=True).to_dict(orient='records')  # Each row becomes a dictionary
# with open('C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-\\initialization\\transformed_df.json', 'w') as json_file:
#     json.dump(df_dict, json_file, indent=4)