''' In this code we stored the functions that were used in the cleaning section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt 
from collections import OrderedDict
from dateutil import parser
from datetime import timezone
from collections import deque
from information import features, identity, kpi, infoManager

''''
________________________________________________________________________________________________________
FUNCTIONS FOR DATA CLEANING
________________________________________________________________________________________________________
'''


# ______________________________________________________________________________________________
# This function takes in input the data point that we are receiving and checks the reliability 
# of its features in terms of logic consistency (min<=avg<=max<=sum). If one of these conditions is 
# not satisfied, then it means that the involved features are not working as expected. In this function
# we set the corrisponding indicator as False (check not passed), and in the main code (validate_format) the
# corrisponding value will be put at nan since its information is not reliable.

def check_f_consistency(x):
    indicator=[True, True, True, True]
    if not np.isnan(x['min']) and not np.isnan(x['avg']):
        if x['min'] > x['avg']:
            indicator[2]=False
            indicator[1]=False
    if not np.isnan(x['min']) and not np.isnan(x['max']):
        if x['min'] > x['max']:
            indicator[2]=False
            indicator[3]=False
    if not np.isnan(x['min']) and not np.isnan(x['sum']):
        if x['min'] > x['sum']:
            indicator[2]=False
            indicator[0]=False
    if not np.isnan(x['avg']) and not np.isnan(x['max']):
        if x['avg'] > x['max']:
            indicator[1]=False
            indicator[3]=False
    if not np.isnan(x['avg']) and not np.isnan(x['sum']):
        if x['avg'] > x['sum']:
            indicator[1]=False
            indicator[0]=False
    if not np.isnan(x['max']) and not np.isnan(x['sum']):
        if x['max'] > x['sum']:
            indicator[0]=False
            indicator[3]=False
    return indicator

# ______________________________________________________________________________________________
# This function takes in input the data point that we are receiving and checks its reliability in
# terms of format. In general, if the data point is too severly compromised (one of the identity fields is 
# nan or missing, all features are nan), then it is discarded (return None).

def validate_format(x):

    missing_identity = [field for field in identity if field not in list(x.keys())]
    missing_features = [field for field in features if field not in list(x.keys())]

    # Check is any identity field is missing or if any of them is nan.
    if missing_identity or any(pd.isna(x.get(key)) for key in identity):
        print('Data point misses essential fields. Discarded.')
        return None # In this case the data point is discarder: its identity is unknown.
    # Check if all the features that the datapoint has are nan or missing.
    elif all(pd.isna(x.get(key)) for key in features):
        print('Data point is useless becasue either all features are nan or missing. Discarded')
        return None # Also in this case the data point is discarded since it doesn't even contain the least meaningful information.
    else: # It means that the identity is well defined and at least one of the feature values is non nan --> the data point can be useful.
        for mf in missing_features:
            x[mf] = np.nan

    x = dict(OrderedDict((key, x[key]) for key in identity + features))

    # Try to set the format of the 'time' field into the most used one (ISO 8601 format in UTC)
    try:
        date_obj = parser.parse(x['time'])
        x['time'] = date_obj.replace(tzinfo=timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    except Exception as e:
        print("Invalid time format. Discarded data point.")
        return None
    
    # Check if the features (min, max, sum, avg) sarisfy the basic logic rule min<=avg<=max<=sum
    cc=check_f_consistency(x)
    if not any(cc): #meaning that no feature respect the logic rule
        return None #discard the datapoint: too much compromised.
    else:
        for f, c in zip(features, cc):
            if c==False:
                x[f]=np.nan
    return x

# ______________________________________________________________________________________________
# This function is an exemplified version of a set of checks regarding the range that the data 
# point value can assume. In first analysis we have considered the kpi to have an expected range
# that is common to all machines. Further improvement may involve define expected ranges specific
# for the machine_type. It will return True if the range is appropriate (passed check) and False 
# otherwise.

def check_range(x):
    flag=True
    if x: # Check the range only if the validation has not discarded the data point.
        l_thr=kpi[x['kpi']][0]
        h_thr=kpi[x['kpi']][1]
        for k in list(x.keys())[-5:] :
            if x[k]<l_thr:
                x[k]=np.nan
                flag=False
        if x['max']>h_thr:
            x['max']=np.nan
            flag=False
        return x, flag
    
# ______________________________________________________________________________________________
# This function is the one that phisically make the imputation for a specific feature of the data point. 
# It receives in input the univariate batch that needs to use and according to the required number of data
# needed by the Exponential Smoothing, it decides to use it or to simply adopt the maean.

def predict_missing(batch):
    seasonality=7
    cleaned_batch= [x for x in batch if not np.isnan(x)]
    #print(cleaned_batch)
    if not(all(pd.isna(x) for x in batch)) and batch:
        if len(cleaned_batch)>2*seasonality:
            print('**Use Exp Smoothing for prediction**')
            model = ExponentialSmoothing(cleaned_batch, seasonal='add', trend='add', seasonal_periods=seasonality)
            model_fit = model.fit()
            prediction = model_fit.forecast(steps=1)[0]
        else:
            print('**Use mean for prediction**')
            prediction=np.nanmean(batch)
        return prediction
    else: 
        return np.nan # Leave the feature as nan since we don't have any information in the batch to make the imputation.

# ______________________________________________________________________________________________
# This function is the one managing the imputation for all the features of the data point  receives as an input the new data point, extracts the information

def imputer(x):
    if x:
        x=x[0] #Because checked data will return two values (the data point and the result of the check)
        nan_cons_thr=3
        im=infoManager()
        
        # Try imputation with mean or the HWES model.
        for f in features:
            batch = im.get_batch(f)
            if pd.isna(x[f]):
                    counter=im.update_counter(f)
                    x[f]=predict_missing(batch)
            else: 
                counter=im.update_counter(f, True)
            if counter>nan_cons_thr:
                point_id='/'.join(map(str, list(x.values())[1:4]+list([f])))
                print("It's been " + str(counter) + ' days that [' + str(point_id) + '] is missing')
        print(f'after imputation dp: {x}')

        # Check again the consistency of features and the range.
        if check_f_consistency(x) and check_range(x)[1]:
            pass
        else:  # It means that the imputed data point has not passed the check on the features and on their expected range.
            # In this case we use the LVCF as a method of imputation since it ensures the respect of these conditiono (the last point in the batch has been preiovusly checked)
            for f in features:
                batch=im.get_batch(f)
                x[f]=batch[-1]

        print(f'after check again dp: {x}')
        
        # In the end update batches with the new data point
        for f in features:
            print(f'original batch {f}: {im.get_batch(f)}')
            im.update_batch(f, x[f])
            print(f'batch after update {f}: {im.get_batch(f)}')

        return x
