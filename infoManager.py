from datetime import timezone
from collections import deque
import json

identity=['asset_id', 'name', 'kpi', 'operation']
features=['sum', 'avg', 'min', 'max', 'var']
store_path="store.json"
discarded_path='discarded_dp.json'
data_path='synthetic_data.json'
b_length=40
faulty_aq_tol=3

machine={'metal_cutting': ['ast-yhccl1zjue2t', 'ast-ha448od5d6bd', 'ast-6votor3o4i9l', 'ast-5aggxyk5hb36', 'ast-anxkweo01vv2', 'ast-6nv7viesiao7'],
'laser_cutting': ['ast-xpimckaf3dlf'],
'laser_welding': ['ast-hnsa8phk2nay', 'ast-206phi0b9v6p'],
'assembly': ['ast-pwpbba0ewprp', 'ast-upqd50xg79ir', 'ast-sfio4727eub0'],
'testing': ['ast-nrd4vl07sffd', 'ast-pu7dfrxjf2ms', 'ast-06kbod797nnp'],
'riveting': ['ast-o8xtn5xa8y87']}

ML_algorithms_config = {
    'forecasting_ffnn': {
        'make_stationary': True,  # Default: False
        'detrend': False,          # Default: False
        'deseasonalize': False,    # Default: False
        'get_residuals': False,    # Default: False
        'scaler': True             # Default: True
    },
    'anomaly_detection': {
        'make_stationary': False, # Default: False
        'detrend': False,         # Default: False
        'deseasonalize': False,   # Default: False
        'get_residuals': False,    # Default: False
        'scaler': False           # Default: False
    }
}

# The following dictionary is organized as follows: for each type of kpi [key], the corrisponding value is a list of two elements - min and max of the expected range.
# We consider in this dictionary only 'pure' kpis that we expect from machines directly, as indicated in the tassonomy produced by the topic 1.
kpi={'time': [[0, 86400], ['working', 'idle', 'offline']], # As indicated in the taxonomy the time is reported in seconds.
     'consumption': [[0, 500000], ['working', 'idle', 'offline']], #KWh
     'power': [[0, 200000], ['independent']], #KW
     'emission_factor': [[0, 3],['independent']], #kg/kWh
     'cycles': [[0, 300000], ['working']], #number
     'average_cycle_time': [[0, 4000],['working']], #seconds
     'good_cycles': [[0, 300000],['working']], #number
     'bad_cycles': [[0, 300000],['working']], #number 
     'cost': [[0, 1],['independent']] #euro/kWh
     }


def get_batch(x, f):    
    with open(store_path, "r") as json_file:
            info = json.load(json_file)    
    # This function will return batch
    return list(info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])



def update_batch(x, f, p): 
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    dq=deque(info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])
    dq.append(p)
    
    if len(dq)>b_length:
        dq.popleft()
    # Store the new batch into the info dictionary.
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)]=dq

    with open(store_path, "w") as json_file:
        json.dump(info, json_file, indent=1) 



def update_counter(x, reset=False):
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    if reset==False:
        info[x['name']][x['asset_id']][x['kpi']][x['operation']][1]=info[x['name']][x['asset_id']][x['kpi']][x['operation']][1]+1
    else:
        info[x['name']][x['asset_id']][x['kpi']][x['operation']][1]=0
    
    with open(store_path, "w") as json_file:
        json.dump(info, json_file, indent=1) 



def get_counter(x):
    with open(store_path, "r") as json_file:
        info = json.load(json_file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][1]



def get_model_ad(x): #id should contain the identity of the kpi about whihc we are storing the model 
                           #[it is extracted from the columns of historical data, so we expect it to be: asset_id, name, kpi, operation]
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][2]



def update_model_ad(x, model):
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][2]=model

    with open(store_path, "w") as json_file:
        json.dump(info, json_file, indent=1) 



def get_model_forecast(x): #id should contain the identity of the kpi about whihc we are storing the model                        #[it is extracted from the columns of historical data, so we expect it to be: asset_id, name, kpi, operation]
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][3]



def update_model_forecast(x, model):
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][3]=model
    
    with open(store_path, "w") as json_file:
        json.dump(info, json_file, indent=1) 

