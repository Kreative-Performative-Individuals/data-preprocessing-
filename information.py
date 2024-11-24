from collections import OrderedDict
from dateutil import parser
from datetime import timezone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from collections import OrderedDict
from dateutil import parser
from datetime import timezone
from collections import deque


identity=['time', 'asset_id', 'name', 'kpi']
features=['sum', 'avg', 'min', 'max', 'var']

machine={'metal_cutting': ['ast-yhccl1zjue2t', 'ast-ha448od5d6bd', 'ast-6votor3o4i9l', 'ast-5aggxyk5hb36', 'ast-anxkweo01vv2', 'ast-6nv7viesiao7'],
'laser_cutting': ['ast-xpimckaf3dlf'],
'laser_welding': ['ast-hnsa8phk2nay', 'ast-206phi0b9v6p'],
'assembly': ['ast-pwpbba0ewprp', 'ast-upqd50xg79ir', 'ast-sfio4727eub0'],
'testing': ['ast-nrd4vl07sffd', 'ast-pu7dfrxjf2ms', 'ast-06kbod797nnp'],
'riveting': ['ast-o8xtn5xa8y87']}

# The following dictionary is organized as follows: for each type of kpi [key], the corrisponding value is a list of two elements - min and max of the expected range.
kpi={'energy_cost': [0, 1000], 
     'cost_per_unit': [0, 1000],
     'working_time': [0, 24], # Assuming that the information is reported in hours.
     'idle_time': [0, 24], # //
     'offline_time': [0, 24], # //
     'total_downtime_duration': [0, 24], #//
     'mtbf': [0, 24], #//
     'cycles_count': [0, 10000],
     'average_cycle_time': [0, 60], # Assumed to be expressed in minutes. 
     'good_cycle_count': [0, 10000],
     'bad_cycle_count': [0, 10000],
     'machine_working_consumption': [0, 100],
     'machine_idle_consumption': [0, 100],
     'carbon_foot_print': [0, 20]}

class infoManager:
    def __init__(self, x):
        self.x=x
        self.machine=x['name']
        self.id=x['asset_id']
        self.kpi=x['kpi']
        self.b_length=40 # It can be changed

    def get_batch(self, f):        
        # This function will return batch, counter
        return list(info[self.machine][self.id][self.kpi][0][features.index(f)])
    
    def update_batch(self, f, p): 
        dq=deque(info[self.machine][self.id][self.kpi][0][features.index(f)])
        dq.append(p)
        
        if len(dq)>self.b_length:
            dq.popleft()
        # Store the new batch into the info dictionary.
        info[self.machine][self.id][self.kpi][0][features.index(f)]=dq
    
    def update_counter(self, f, reset=False):
        if reset==False:
            info[self.machine][self.id][self.kpi][1][features.index(f)]=info[self.machine][self.id][self.kpi][1][features.index(f)]+1
        else:
            info[self.machine][self.id][self.kpi][1][features.index(f)]=0
        return info[self.machine][self.id][self.kpi][1][features.index(f)] 