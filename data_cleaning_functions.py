''' In this code we stored the functions that were used in the cleaning section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

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

''''
________________________________________________________________________________________________________
FUNCTIONS FOR DATA CLEANING
________________________________________________________________________________________________________
'''
# ______________________________________________________________________________________________
# This function takes in input the data point that we are receiving and checks its reliability
# in terms of logic consistency, so it checks if the features (min, max, sum, avg) sarisfy the
# basic logic rule min<=avg<=max<=sum. In output it will return an indicator array, where each
# cell gives False if the rule is not respected or True if it is.

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
# terms of format. In output it will return nothing if the data point is too incomplete to 
# correctly identify and use it, or the reconstructed data point with nans whenever some value is
# missing.

def validate_format(x):
    # Check if all essential columns are present: if the first four fields are missing then
    # the identity of the data point is unknown, thus, it needs to be deleted. 
    # Otherwise, (one of the remaining 4 fields in missing), set the missing column as missing. 
    # If all the aggregates are missing the data point is discarded as well.

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
# point value can assume according to the specific physical quantity that it represents. It 
# receives as an input the input data point and checks it ranges according to the kpi information.
# If the value is outside given thresholds if returns False, if everything is okey returns True.

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
# This function receives as an input a batch of data representing the previous data points. 
# It applies the model Exponential Smoothing to predict and return one future step in the time 
# serie or None if there is no sufficient information in the batch to fill the NaN value.

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
# This function receives as an input the new data point, extracts the information

def imputer(x):
    if x:
        x=x[0]
        nan_cons_thr=3
        im=infoManager(x)

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
