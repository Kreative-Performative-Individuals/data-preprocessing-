import numpy as np
import pandas as pd
from datetime import timezone
from collections import deque


identity=['time', 'asset_id', 'name', 'kpi', 'operation']
features=['sum', 'avg', 'min', 'max', 'var']

machine={'metal_cutting': ['ast-yhccl1zjue2t', 'ast-ha448od5d6bd', 'ast-6votor3o4i9l', 'ast-5aggxyk5hb36', 'ast-anxkweo01vv2', 'ast-6nv7viesiao7'],
'laser_cutting': ['ast-xpimckaf3dlf'],
'laser_welding': ['ast-hnsa8phk2nay', 'ast-206phi0b9v6p'],
'assembly': ['ast-pwpbba0ewprp', 'ast-upqd50xg79ir', 'ast-sfio4727eub0'],
'testing': ['ast-nrd4vl07sffd', 'ast-pu7dfrxjf2ms', 'ast-06kbod797nnp'],
'riveting': ['ast-o8xtn5xa8y87']}

ML_algorithms_config = {
    'forecasting_ffnn': {
        'make_stationary': True,  # Default: False
        'detrend': True,          # Default: False
        'deseasonalize': True,    # Default: False
        'get_residuals': True,    # Default: False
        'scaler': True             # Default: True
    },
    'anomaly_detection': {
        'make_stationary': False, # Default: False
        'detrend': False,         # Default: False
        'deseasonalize': False,   # Default: False
        'get_residuals': True,    # Default: False
        'scaler': False           # Default: True
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


class infoManager:
    def __init__(self, info):
        self.info=info

    def get_batch(self, x, f):        
        # This function will return batch, counter
        return list(self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])
    
    def update_batch(self, x, f, p): 
        b_length=40
        dq=deque(self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])
        dq.append(p)
        
        if len(dq)>b_length:
            dq.popleft()
        # Store the new batch into the info dictionary.
        self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)]=dq
    
    def update_counter(self, x, f, reset=False):
        if reset==False:
            self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][1][features.index(f)]=self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][1][features.index(f)]+1
        else:
            self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][1][features.index(f)]=0
        return self.info[x['name']][x['asset_id']][x['kpi']][x['operation']][1][features.index(f)]

    def get_model_ad(self, id): #id should contain the identity of the kpi about whihc we are storing the model 
                                     #[it is extracted from the columns of historical data, so we expect it to be: asset_id, name, kpi, operation]
         return self.info[id[1]][id[0]][id[2]][id[3]][2]
    
    def update_model_ad(self, id, model):
        self.info[id[1]][id[0]][id[2]][id[3]][2]=model
