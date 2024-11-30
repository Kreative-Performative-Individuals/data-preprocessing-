''' In this code we stored the functions that were used in the cleaning section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from collections import OrderedDict
from dateutil import parser
from datetime import timezone, datetime
from infoManager import features, identity, kpi, faulty_aq_tol, get_batch, update_counter, update_batch, get_counter, discarded_path
import json

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

def save_disc_dp(x):
    with open(discarded_path, "r") as json_file:
        discarded_dp = json.load(json_file)
    with open(discarded_path, "w") as json_file:
        json.dump(discarded_dp.append(x), json_file, indent=1) 

def validate(x):
    missing_identity = [field for field in identity if field not in list(x.keys())]
    missing_features = [field for field in features if field not in list(x.keys())]

    # Check is any identity field is missing or if any of them is nan.
    if missing_identity or any(pd.isna(x.get(key)) for key in identity + ['time']):
        time=x['time']
        print(f'Data point at time {time} misses essential fields. Discarded.')
        update_counter(x)
        save_disc_dp(x)
        return None # In this case the data point is discarder: its identity is unknown.
    # Check if all the features that the datapoint has are nan or missing.
    elif all(pd.isna(x.get(key)) for key in features):
        print(f'Data point {time} is useless becasue either all features are nan or missing. Discarded')
        update_counter(x)
        save_disc_dp(x)
        return None # Also in this case the data point is discarded since it doesn't even contain the least meaningful information.
        
    else: # It means that the identity is well defined and at least one of the feature values is non nan --> the data point can be useful.
        for mf in missing_features:
            x[mf] = np.nan

    x = dict(OrderedDict((key, x[key]) for key in identity + features))

    # Try to transform the timestamp into datetime
    try:
        date_obj = parser.parse(x['time'])
        x['time'] = date_obj.replace(tzinfo=timezone.utc)
    except Exception as e:
        update_counter(x)
        save_disc_dp(x)
        print("Invalid time format. Discarded data point.")
        return None
    
    x, _=check_range(x)
    
    # Check if the features (min, max, sum, avg) satisfy the basic logic rule min<=avg<=max<=sum
    cc=check_f_consistency(x)
    if all(cc==False): #meaning that no feature respect the logic rule
        print(f'The data point at time {time} is too much compromised. Discarded')
        update_counter(x)
        save_disc_dp(x)
        return None #discard the datapoint: too much compromised.
    elif any(cc==False): #at least one feature value doesn't behave as it should
        update_counter(x)
        save_disc_dp(x)
        for f, c in zip(features, cc):
            if c==False:
                x[f]=np.nan
    #print(f'data point after validation: {x}')
    # if the data points arrives till here it means that it is ok from the format, logical and range point of view --> if it has nans it means that it has meet some non serious problems that could have lead to the discard.
    if any(np.isnan(value) for value in [x.get(key) for key in features]):
        update_counter(x)
    else: 
        #it means that the data point is perfect
        update_counter(x, True)
    return x

def check_range(x):
    # Check range
    flag=True

    l_thr=kpi[x['kpi']][0][0]
    h_thr=kpi[x['kpi']][0][1]
    for k in [x.get(key) for key in features]:
        if x[k]<l_thr:
            x[k]=np.nan
            flag=False
        if k in ['avg', 'max', 'min'] and x[k]>h_thr:
            x[k]=np.nan
            flag=False
    # if after checking the range all features are nan, discard.
    if all(np.isnan(value) for value in [x.get(key) for key in features]):
        update_counter(x)
        save_disc_dp(x)
        return None
    else:
        return x, flag
    
    #in this function, if the data is invalid the fact that it returns None is an indicator itself.


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
            #print('**Use Exp Smoothing for prediction**')
            model = ExponentialSmoothing(cleaned_batch, seasonal='add', trend='add', seasonal_periods=seasonality)
            model_fit = model.fit()
            prediction = model_fit.forecast(steps=1)[0]
        else:
            #print('**Use mean for prediction**')
            prediction=np.nanmean(batch)
        return prediction
    else: 
        return np.nan # Leave the feature as nan since we don't have any information in the batch to make the imputation.

# ______________________________________________________________________________________________
# This function is the one managing the imputation for all the features of the data point  receives as an input the new data point, extracts the information

def imputer(x):
    if x:
        x=x[0] #Because checked data will return two values (the data point and the result of the check)

        # Try imputation with mean or the HWES model.
        for f in features:
            batch = get_batch(x, f)
            if pd.isna(x[f]):
                    x[f]=predict_missing(batch)
        #print(f'after imputation dp: {x}')

        # Check again the consistency of features and the range.
        if check_f_consistency(x) and check_range(x)[1]:
            pass
        else:  # It means that the imputed data point has not passed the check on the features and on their expected range.
            # In this case we use the LVCF as a method of imputation since it ensures the respect of these conditiono (the last point in the batch has been preiovusly checked)
            for f in features:
                batch = get_batch(x, f)
                x[f]=batch[-1]

        #print(f'after check again dp: {x}')
        
        # In the end update batches with the new data point
        for f in features:
            #print(f'original batch {f}: {im.get_batch(x, f)}')
            update_batch(x, f, x[f])
            #print(f'batch after update {f}: {im.get_batch(x, f)}')

        return x

# ______________________________________________________________________________________________
# This function implements all the steps needed for the cleaning in order to fuse the cleaning into one code line.
def cleaning_pipeline(x):
    old_counter=get_counter(x)
    validated_dp=validate(x)
    new_counter=get_counter(x)
    if new_counter==old_counter+1 and new_counter>=faulty_aq_tol:
        id = {key: x[key] for key in identity if key in x}
        print(f"it has been {new_counter} days up to now that {id} reports problem in the acquisition! Check it out!") #generate the alert: it has been new_counter days that x[identity] has problems with the acquisition.
    cleaned_dp=imputer(validated_dp)

    return cleaned_dp

#test with this:
# dp={
#     'time': datetime.now(),
#     'asset_id':  'ast-yhccl1zjue2t',
#     'name': 'metal_cutting',
#     'kpi': 'time',
#     'operation': 'working',
#     'sum': 10,
#     'avg': 3,
#     'min': 1,
#     'max': np.nan,
#     'var': 4}
#it should alert that there is a problem in the acquisition but the point is still cleaned.