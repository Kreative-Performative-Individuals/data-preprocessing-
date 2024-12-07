''' In this code we stored the functions that were used in the data processing pipeline,
including a brief description of their inputs, outputs and functioning'''


import numpy as np
import pandas as pd
import json
from collections import OrderedDict, deque
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from river import drift
import optuna
from connections_functions import send_alert, store_datapoint
import pickle
import os


''''
________________________________________________________________________________________________________
FUNCTIONS FOR INFO MANAGER
________________________________________________________________________________________________________
'''
''' In this code we stored the functions that were used in the info manager section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

fields = ['time', 'asset_id', 'name', 'kpi', 'operation', 'sum', 'avg', 'min', 'max', 'var']
identity=['asset_id', 'name', 'kpi', 'operation']
features=['sum', 'avg', 'min', 'max', 'var']
store_path = "store.pkl"
store_path_forecasting = "forecasting_models.pkl"
data_path = "synthetic_data.json"
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
    with open(store_path, "rb") as file:
            info = pickle.load(file)    
    # This function will return batch
    return list(info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])



def update_batch(x, f, p): 
    with open(store_path, "rb") as file:
            info = pickle.load(file)
    dq=deque(info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)])
    dq.append(p)
    
    if len(dq)>b_length:
        dq.popleft()
    # Store the new batch into the info dictionary.
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][0][features.index(f)]= list(dq)

    with open(store_path, "wb") as file:
        pickle.dump(info, file) 



def update_counter(x, reset=False):
    with open(store_path, "rb") as file:
            info = pickle.load(file)
    if not reset:
        info[x['name']][x['asset_id']][x['kpi']][x['operation']][1] += 1
    else:
        info[x['name']][x['asset_id']][x['kpi']][x['operation']][1] = 0
    
    with open(store_path, "wb") as file:
        pickle.dump(info, file) 



def get_counter(x):
    with open(store_path, "rb") as file:
        info = pickle.load(file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][1]



def get_model_ad(x): #id should contain the identity of the kpi about whihc we are storing the model 
                           #[it is extracted from the columns of historical data, so we expect it to be: asset_id, name, kpi, operation]
    with open(store_path, "rb") as file:
            info = pickle.load(file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][2]



def update_model_ad(x, model):
    with open(store_path, "rb") as file:
            info = pickle.load(file)
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][2]=model

    with open(store_path, "wb") as file:
        pickle.dump(info, file) 



'''def get_model_forecast(x): #id should contain the identity of the kpi about whihc we are storing the model                        #[it is extracted from the columns of historical data, so we expect it to be: asset_id, name, kpi, operation]
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    return info[x['name']][x['asset_id']][x['kpi']][x['operation']][3]'''

def get_model_forecast(x):
    # Load the Pickle file
    with open(store_path_forecasting, "rb") as f:
        all_models_data = pickle.load(f)
    
    # Navigate to the specific model based on the x keys
    model_info = all_models_data[x['name']][x['asset_id']][x['kpi']][x['operation']]

    models_for_subfeatures = {}
    for sub_feature, data in model_info.items():
        keras_model = model_from_json(data['model_architecture'])
        keras_model.set_weights(data['model_weights'])
        models_for_subfeatures[sub_feature] = [keras_model, data['params'], data['stats']]
    
    return models_for_subfeatures

'''def update_model_forecast(x, model):
    with open(store_path, "r") as json_file:
            info = json.load(json_file)
    info[x['name']][x['asset_id']][x['kpi']][x['operation']][3]=model
    
    with open(store_path, "w") as json_file:
        json.dump(info, json_file, indent=1) '''

def update_model_forecast(x, model):
    """
    Update or store models, parameters, and stats in a Pickle file.
    
    Arguments:
    - x: A dictionary that contains keys to locate the entry (e.g., {'name', 'asset_id', 'kpi', 'operation'}).
    - model: A dictionary where the keys are sub-features ('min', 'max', 'sum', 'avg') and 
             the values are lists containing [keras_model, best_params, stats].
    - store_path: The path where the Pickle file is stored.
    
    This function will either add new models or update existing ones in the Pickle file.
    """
    
    # Load the existing data from the Pickle file if it exists
    if os.path.exists(store_path_forecasting):
        with open(store_path_forecasting, "rb") as f:
            all_models_data = pickle.load(f)
    else:
        all_models_data = {}

    # Navigate to the specific location in the dictionary based on 'x' keys
    name = x['name']
    asset_id = x['asset_id']
    kpi = x['kpi']
    operation = x['operation']
    
    # Check if the structure exists, if not, initialize it
    if name not in all_models_data:
        all_models_data[name] = {}
    
    if asset_id not in all_models_data[name]:
        all_models_data[name][asset_id] = {}
    
    if kpi not in all_models_data[name][asset_id]:
        all_models_data[name][asset_id][kpi] = {}
    
    if operation not in all_models_data[name][asset_id][kpi]:
        all_models_data[name][asset_id][kpi][operation] = {}

    # Now, for each sub-feature, add or update the model, params, and stats
    for sub_feature, model_info in model.items():
        keras_model, best_params, stats = model_info
        
        # Store model data in the dictionary
        all_models_data[name][asset_id][kpi][operation][sub_feature] = {
            'model_architecture': keras_model.to_json(),  # Store model architecture
            'model_weights': keras_model.get_weights(),   # Store model weights
            'params': best_params,                        # Store best parameters
            'stats': stats                                # Store stats
        }
    
    # Save the updated data back to the Pickle file
    with open(store_path_forecasting, "wb") as f:
        pickle.dump(all_models_data, f)

    print(f"Models updated successfully in {store_path_forecasting}")

''''
________________________________________________________________________________________________________
FUNCTIONS FOR DATA CLEANING
________________________________________________________________________________________________________
'''
''' In this code we stored the functions that were used in the cleaning section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

# ______________________________________________________________________________________________
# This function takes in input the data point that we are receiving and checks the reliability 
# of its features in terms of logic consistency (min<=avg<=max<=sum). If one of these conditions is 
# not satisfied, then it means that the involved features are not working as expected. In this function
# we set the corrisponding indicator as False (check not passed), and in the main code (validate_format) the
# corrisponding value will be put at nan since its information is not reliable.

def check_f_consistency(x):
    indicator=[True, True, True, True]
    if not pd.isna(x['min']) and not pd.isna(x['avg']):
        if x['min'] > x['avg']:
            indicator[2]=False
            indicator[1]=False
    if not pd.isna(x['min']) and not pd.isna(['max']):
        if x['min'] > x['max']:
            indicator[2]=False
            indicator[3]=False
    if not pd.isna(x['min']) and not pd.isna(x['sum']):
        if x['min'] > x['sum']:
            indicator[2]=False
            indicator[0]=False
    if not pd.isna(x['avg']) and not pd.isna(x['max']):
        if x['avg'] > x['max']:
            indicator[1]=False
            indicator[3]=False
    if not pd.isna(x['avg']) and not pd.isna(x['sum']):
        if x['avg'] > x['sum']:
            indicator[1]=False
            indicator[0]=False
    if not pd.isna(x['max']) and not pd.isna(x['sum']):
        if x['max'] > x['sum']:
            indicator[0]=False
            indicator[3]=False
    if pd.isna(x['sum']):
        indicator[0]=False
    if pd.isna(x['avg']):
        indicator[1]=False
    if pd.isna(x['min']):
        indicator[2]=False
    if pd.isna(x['max']):
        indicator[3]=False
    return indicator


def validate(x):
    """Takes in input the data point that we are receiving and checks its reliability in
    terms of format. In general, if the data point is too severly compromised (one of the identity fields is 
    nan or missing, all features are nan), then it is discarded (return None).

    Args:
        x (_type_): the data point

    Returns:
        _type_: _description_
    """

    for f in fields:
        x.setdefault(f, np.nan) #if some fields is missing from the expected ones, put a nan
    x = dict(OrderedDict((key, x[key]) for key in fields)) # order the fields of the datapoint

    # Ensure the reliability of the field time
    if pd.isna(x['time']):
        x['time'] = datetime.now()

    # Check that there is no missing information in the identity of the datapoint, otherwise we store in the database, labelled 'Corrupted'.
    if any(pd.isna(x.get(key)) for key in identity):
        #Future developments for updating the counter in this case.
        x['status']='Corrupted'
        store_datapoint(x)
        return None
    # if the code run forward it means that the identity is intact.
    old_counter = get_counter(x)
    #print(f'old counter {old_counter}')
    # Check if all the features that the datapoint has are nan or missing.
    if all(pd.isna(x.get(key)) for key in features):
        update_counter(x)
        x['status']='Corrupted'
        store_datapoint(x)
        return None, old_counter
    
    #if the datapoint comes here it means that it didn't miss any information about the identity and at least one feature that is not nan.

    x=check_range(x) #Here i change letter since i need x to get the counter later.

    #if the datapoint comes here it means that at least one feature value is respecting the range constraint for the specific kpi.
    if x:
        # Check if the features (min, max, sum, avg) satisfy the basic logic rule min<=avg<=max<=sum
        cc=check_f_consistency(x)
        if all(not c for c in cc): #meaning that no feature respect the logic rule
            if pd.isna(x['var']):
                update_counter(x)
                x['status']='Corrupted'
                store_datapoint(x)
                return None, old_counter
        elif all(c for c in cc): #the datapoint verifies the logic rule.
                            #if now there is a nan it could be either the result of the range check or that the datapoint intrinsically has these nans.
            any_nan=False
            for f in features:
                if pd.isna(x[f]):
                    any_nan=True
                    if all(pd.isna(get_batch(x, f))):
                        pass
                    else:
                        update_counter(x)
                        break
            if not any_nan:
                                 #it means that the datapoint is consistent and it doesn't have nan values --> it is perfect.
                update_counter(x, True) #reset the counter.
        else: #it means that some feature are consistent and some not. Put at nan the not consistent ones.
            for f, c in zip(features, cc):
                if not c:
                    x[f]=np.nan
            update_counter(x)
    return x, old_counter



def check_range(x):

    #Retrieve the specific range for the kpi that we are dealing with
    l_thr=kpi[x['kpi']][0][0]
    h_thr=kpi[x['kpi']][0][1]

    for k in features:
        if x[k]<l_thr:
            x[k]=np.nan
        if k in ['avg', 'max', 'min', 'var'] and x[k]>h_thr:
            x[k]=np.nan

    # if after checking the range all features are nan --> corrupted
    if all(pd.isna(value) for value in [x.get(key) for key in features]):
        update_counter(x)
        x['status']='Corrupted'
        store_datapoint(x)
        return None
    else:
        return x

def check_range_ai(x):
    flag=True #takes trace of: has the datapoint passed the range check without being changed?
    l_thr=kpi[x['kpi']][0][0]
    h_thr=kpi[x['kpi']][0][1]

    for k in features:
        if x[k]<l_thr:
            flag=False
        if k in ['avg', 'max', 'min', 'var'] and x[k]>h_thr:
            flag=False
    return flag


# ______________________________________________________________________________________________
# This function is the one that phisically make the imputation for a specific feature of the data point. 
# It receives in input the univariate batch that needs to use and according to the required number of data
# needed by the Exponential Smoothing, it decides to use it or to simply adopt the maean.

def predict_missing(batch):
    seasonality=7
    cleaned_batch= [x for x in batch if not pd.isna(x)]
    if not(all(pd.isna(x) for x in batch)) and batch:
        if len(cleaned_batch)>2*seasonality:
            model = ExponentialSmoothing(cleaned_batch, seasonal='add', trend='add', seasonal_periods=seasonality)
            model_fit = model.fit()
            prediction = model_fit.forecast(steps=1)[0]
        else:
            prediction=float(np.nanmean(batch))
        return prediction
    else: 
        return np.nan # Leave the feature as nan since we don't have any information in the batch to make the imputation. If the datapoint has a nan because the feature is not definable for it, it will be leaved as it is from the imputator.

# ______________________________________________________________________________________________
# This function is the one managing the imputation for all the features of the data point  receives as an input the new data point, extracts the information

def imputer(x):
    if x:
        if isinstance(x, tuple):
            x = x[0]
            #Because the validated datapoint may exit in the check range with 2 returned values.

        # Try imputation with mean or the HWES model.
        for f in features:
            batch = get_batch(x, f)
            if pd.isna(x[f]):
                    x[f]=predict_missing(batch)

        # Check again the consistency of features and the range.
        if check_f_consistency(x) and check_range_ai(x):
            pass
        else:  # It means that the imputed data point has not passed the check on the features and on their expected range.
            # In this case we use the LVCF as a method of imputation since it ensures the respect of these conditiono (the last point in the batch has been preiovusly checked)
            for f in features:
                batch = get_batch(x, f)
                x[f]=batch[-1]
        
        # In the end update batches with the new data point
        for f in features:
            update_batch(x, f, x[f])

        return x

# ______________________________________________________________________________________________
# This function implements all the steps needed for the cleaning in order to fuse the cleaning into one code line.

def cleaning_pipeline(x):
    validated_dp, old_counter=validate(x)
    new_counter=get_counter(x)
    #print(f'new counter: {new_counter}')
    if new_counter==old_counter+1 and new_counter>=faulty_aq_tol:
        id = {key: x[key] for key in identity if key in x}
        send_alert(id, 'Nan', new_counter)
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

''''
________________________________________________________________________________________________________
FUNCTIONS FOR DRIFT DETECTION
________________________________________________________________________________________________________
'''

''' In this code we stored the functions that were used in the drift detection section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

# ______________________________________________________________________________________________
# This function takes in input the time serie specific for a feature of a determined machine and
# KPI. It computes the potential drift points present in the given time range and returnes two 
# arguments, the first takes the value False if no drift was detected or True if there is some 
# drift, while the second returns the drift points. 

def ADWIN_drift(x, delta=0.005, drift_threshold=3):
    """
    Detects concept drift using the ADWIN algorithm.

    Parameters:
        x (DataFrame): Input data.
        features (list): List of feature names to check for drift.
        b_length (int): Minimum required batch length.
        delta (float): Sensitivity parameter for ADWIN (default=0.005).
        drift_threshold (int): Number of feature drifts to signal overall drift (default=3).

    Returns:
        bool: True if overall drift is detected, False otherwise.
    """
    drift_count_per_f = 0

    for f in features:
        # Get the feature data (batch)
        batch = get_batch(x, f)

        if len(batch)<b_length:
            return False
        else:
            # Initialize ADWIN instance for this feature
            adwin = drift.ADWIN(delta=delta)
            # Check for drift
            for i, value in enumerate(batch):
                adwin.update(value)
                if adwin.drift_detected:
                    drift_count_per_f += 1

    if drift_count_per_f>=drift_threshold:
        return True
    else: 
        return False


''''
________________________________________________________________________________________________________
FUNCTIONS FOR ANOMALY DETECTION
________________________________________________________________________________________________________
'''

''' In this code we stored the functions that were used in the anomaly detection section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

# ______________________________________________________________________________________________
# This class is the one responsible for the training and prediction of anomalies. For the training part 
# it will return the trained model for the specific identity; whereas for the prediction part, it will 
# take a single data point in input and return the prediction.


def ad_train(historical_data):
    #account also for the case in which one feature may not be definable for a kpi

    train_set=pd.DataFrame(historical_data)[features]
    nan_columns = train_set.columns[train_set.isna().all()]
    train_set = train_set.drop(columns=nan_columns)
    train_set=train_set.fillna(0)

    s=[]
    cc=np.arange(0.01, 0.5, 0.01)
    for c in cc:
        model = IsolationForest(n_estimators=200, contamination=c)
        an_pred=model.fit_predict(train_set)
        if len(set(an_pred)) > 1:  # Check for multiple clusters
            s.append(silhouette_score(train_set, an_pred))
        else:
            s.append(-1)
    if max(s)<=0.70:
        optimal_c=1e-5
    else:
        optimal_c=cc[np.argmax(s)]
    model = IsolationForest(n_estimators=200, contamination=optimal_c)
    model.fit_predict(train_set)
    return model

def ad_predict(x, model):
    #account for the case in which one feature may be nan also after the imputation since the feature is not definable for that kpi.
    dp=pd.DataFrame.from_dict(x, orient="index").T
    dp=dp[features]
    #nan_columns = dp.columns[dp.isna().all()]
    #dp = dp.drop(columns=nan_columns)
    dp=dp.fillna(0)

    status=model.predict(dp)
    anomaly_score=model.decision_function(dp)
    anomaly_prob=1- (1/(1+np.exp(-5*anomaly_score)))
    anomaly_prob=int(anomaly_prob[0]*100)
    if status==-1:
        status='Anomaly'
    else:
        status='Normal'
    return status, anomaly_prob


''''
________________________________________________________________________________________________________
FUNCTIONS FOR FEATURE ENGINEERING
________________________________________________________________________________________________________
'''
''' In this code we stored the functions that were used in the feature engineering section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

# THIS IS THE MAIN FUNCTION FOR THE FEATURE ENGINEERING
# The input dataframe corresponds to a filtrate version of the dataset for a given machine, kpi and 
# operation, so it contains 9-10  columns (depending on the presence ['sum', 'avg','min', 'max', 'var'])
# and the amount of entries correspondent to the selected time range.
# It also take kwargs as a parameter, which recall the information about how to set the time serie in
# order to have a proper input for the machine learning algortithms.
# It performs several operations on the time series, depending on the kwargs, such as  make_stationary, detrend,
# deseasonalize, get_residuals, scaler.
# It gives as an output the transformed time serie.

def feature_engineering_pipeline(dataframe, kwargs):
    features = ['sum', 'avg','min', 'max']
    for feature_name in features:
        # Check if the column exists in the DataFrame
        if feature_name in dataframe.columns:
            print("-------------------- Results for " + str(feature_name))
            feature = dataframe[feature_name]
            if feature.empty or feature.isna().all() or feature.isnull().all():
                print("Feature is empty (no data).")
            else:
                ## Check stationarity 
                # (output is False if not stationary, True if it is, None if test couldn't be applied)
                is_stationary = adf_test(feature.dropna()) 
                print('Output is stationary? ' + str(is_stationary))
            
                ## Check seasonality
                # (output: period of the seasonality None if no seasonalaty was detected.
                seasonality_period = detect_seasonality_acf(feature)
                print('Seasonality period is? ' + str(seasonality_period))
            
                #further check in the case the seasonality pattern is complex and cannot be detected
                if seasonality_period == None:
                    # (output: period of the seasonality None if no seasonalaty was detected.
                    seasonality_period = detect_seasonality_fft(feature)
                    print('Recomputed seasonality period is? ' + str(seasonality_period))
            
                # (output: the decomposed time series in a list, of form [trend, seasonal, residual],
                # None if it isn't sufficient data or if some error occurs.
                decompositions = seasonal_additive_decomposition(feature, seasonality_period) 

                #Make data stationary / Detrend / Deseasonalize (if needed)
            
                make_stationary = kwargs.get('make_stationary', False)  # Set default to False if not provided
                detrend = kwargs.get('detrend', False) # Set default to False if not provided
                deseasonalize = kwargs.get('deseasonalize', False) # Set default to False if not provided
                get_residuals = kwargs.get('get_residuals', False) # Set default to False if not provided
                scaler = kwargs.get('scaler', False)  # Set default to False if not provided
                
                if make_stationary and (not is_stationary):
                    if decompositions != None:
                        feature = make_stationary_decomp(feature, decompositions)
                        is_stationary = adf_test(feature.dropna())
                        print('Is stationary after trying to make it stationary? ' + str(is_stationary))
                        if not is_stationary:
                            if seasonality_period == None:
                                feature = make_stationary_diff(feature, seasonality_period=[7]) #default weekly
                            else: 
                                feature = make_stationary_diff(feature, seasonality_period=[seasonality_period])
                            is_stationary = adf_test(feature.dropna())
                            print('Is stationary after re-trying to make it stationary? ' + str(is_stationary))
                    else:
                        if seasonality_period == None:
                            feature = make_stationary_diff(feature, seasonality_period=[7]) #default weekly
                        else: 
                            feature = make_stationary_diff(feature, seasonality_period=[seasonality_period])
                        is_stationary = adf_test(feature.dropna())
                        print('Is stationary after trying to make it stationary? ' + str(is_stationary))
            
                if detrend:
                    if decompositions != None:
                        feature = rest_trend(feature, decompositions)
                    else:
                        feature = make_stationary_diff(feature)
                
                if deseasonalize:
                    if decompositions != None:
                        feature = rest_seasonality(feature, decompositions)
                    else:
                        if seasonality_period == None:
                            feature = make_stationary_diff(feature, seasonality_period=[7]) #default weekly
                        else: 
                            feature = make_stationary_diff(feature, seasonality_period=[seasonality_period])
                if get_residuals:
                    if decompositions != None:
                        feature = get_residuals(feature, decompositions)
                    else:
                        feature = make_stationary_diff(feature)
                        if seasonality_period == None:
                            feature = make_stationary_diff(feature, seasonality_period=[7]) #default weekly
                        else: 
                            feature = make_stationary_diff(feature, seasonality_period=[seasonality_period])
                
                if scaler:
                    # Apply standardization (z-score scaling)
                    feature = (feature - np.mean(feature)) / np.std(feature)
            
            dataframe[feature_name] = feature
    
    result_dataframe = dataframe[['time']+features]

    return result_dataframe
# ______________________________________________________________________________________________
# This function takes in input the kpi_name, machine_name, operation_name and the data and filter
# the dataset for the given parameters. It returns the filtered data.

def extract_features(kpi_name, machine_name, operation_name, data):

  filtered_data = data[(data["name"] == machine_name) & (data["kpi"] == kpi_name) & (data["operation"] == operation_name)]

  filtered_data['time'] = pd.to_datetime(filtered_data['time'])
  filtered_data = filtered_data.sort_values(by='time')

  return filtered_data

# ______________________________________________________________________________________________
# This function performs the Augmented Dickey-Fuller test, so it receives as an input
# the time series (it can have nan values, so they need to be filled before) and return
# False if the serie is not stationary and  True if it is (based on the p-value computed
# in the ADF statistics, appliying a statidtical hypothesis test with alfa = 0.05 to
# decide wheter reject or not the null hypothesis (time serie is stationary)). If the 
# series is empty or too short, it return None, indicating that the test couldn't be applied.

def adf_test(series):
    
    if series.empty or len(series) < 2:
        #print("Series is empty or too short for ADF test.")
        return False  # Consider it non-stationary due to insufficient data
    
    try:
        result = adfuller(series)
        if result[1] > 0.05:
                stationarity = False
                #print("The time series is likely non-stationary.")
        else:
                stationarity = True
                #print("The time series is likely stationary.")

        #print(f"ADF Statistic: {result[0]}")
        #print(f"p-value: {result[1]}")
        #print(f"Critical Values: {result[4]}")

        return stationarity
    
    except Exception as e:
        #print(f"Error running ADF test: {e}")
        return None  # If error occurs, consider it non-stationary


# ______________________________________________________________________________________________
# This function allows to detect the seasonality of a time serie. It receive as an input the 
# time series itself, the maximum lag that will be analized (for default) and a threshold that
# refers to the minimum correlation threshold to consider the ACF as significant (default 0.2).
# It applies the ACF at incremental lags and store the significant ones, then it reorder them
# so the first value is the more significant (representing the more prominent seasonality). It 
# returns the highest ACF lag (period of the seasonality) or it returns None if no seasonalaty
# was detected.

def detect_seasonality_acf(df, max_lags=365, threshold=0.2):
    
    # Calculate ACF
    acf_values = acf(df, nlags=max_lags, fft=True)
    
    # Find lags where ACF > threshold (indicating potential seasonality)
    significant_lags = np.where(acf_values[1:] > threshold)[0] + 1  # Find lags with ACF > threshold
    
    if len(significant_lags) == 0:
        return None  # No significant seasonality detected
    
    # Find the lag with the highest ACF value (most prominent)
    highest_acf_lag = significant_lags[np.argmax(acf_values[significant_lags])]

    if highest_acf_lag  <= 1 or highest_acf_lag == len(df):
        return None  # No significant seasonality detected
    
    # Return the corresponding period (seasonality)
    return int(highest_acf_lag)


# ______________________________________________________________________________________________
# This function allows to detect the seasonality of a time serie. It receive as an input the 
# time series itself, apply the FT to detect the maximal peak on frequency that will determine
# a periodicity in the frequency pattern of the signal and return the period corresponding to it.
# If it returns None it's because no seasonalaty was detected.

def detect_seasonality_fft(df):
    
    # Perform FFT
    fft_values = np.fft.fft(df.values)
    
    # Compute the magnitude of the FFT
    fft_magnitude = np.abs(fft_values)
    
    # Ignore the zero frequency (DC component)
    fft_magnitude[0] = 0
    
    # Find the frequency with the highest magnitude
    peak_frequency_index = np.argmax(fft_magnitude)
    peak_frequency = np.fft.fftfreq(len(df), d=1)[peak_frequency_index]
    
    # Convert the frequency to the corresponding period (seasonality period)
    if peak_frequency != 0:
        period = int(round(1 / peak_frequency))
        if period  <= 1 or period == len(df):
            return None  # No significant seasonality detected
    else:
        period = None  # No significant seasonality detected
    
    return period


# ______________________________________________________________________________________________
# This function performs the decomposition  of the time serie into its trend, season
# and residual components (whenever it's possible to implement the analysis). It returns 
# the decomposed time series in a list, of form [trend, seasonal, residual], unless
# there isn't sufficient data or if some error occurs, in that case it returns None.

def seasonal_additive_decomposition(dataframe, period):
    # Check if the filtered DataFrame has enough data for the decomposition
    if dataframe.empty:
        #print(f"No data found for the time serie. Skipping decomposition.")
        return None  

    # Drop NaN values and check if there are enough observations
    series = dataframe.dropna()

    if len(series) < 2:  # Check if the series has at least 2 observations
        print(f"Not enough data. Skipping decomposition.")
        return None  

    if period == None:
        period = 7

    if len(series) < 2 * period:  # Ensure enough data points for at least two full cycles
        print(f"Not enough data for two full cycles. Skipping decomposition.")
        return None  

    # Classical decomposition (additive model)
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)

        #Plot the decomposition
        #plt.figure(figsize=(10, 8))
       # decomposition.plot()
        #plt.suptitle(f'Classical Decomposition of Time Series', fontsize=16)
        #plt.show()

        # Access the individual components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        return [[trend, seasonal, residual]]

    except ValueError as e:
        #print(f"Error during decomposition: {e}")
        return None  



# ______________________________________________________________________________________________
# This function allows to make a time series stationary whenever it is not. It receives as an 
# input the dataserie itself and the computed decompositions (trend, seasonality and residual),
# allowing multiple seasonality analysis. The function rests the trends and seasons sequentially
# and returns a single stationary time series that should be stationary. 

def make_stationary_decomp(df, decompositions):
    # Initialize the stationary series with the original data
    stationary_series = df.copy()

    # Subtract seasonal and trend components from the original data for each seasonality
    for decomposition in decompositions:
        trend = decomposition[0]
        seasonal = decomposition[1]
        
        # Fill NaN values in the trend with the original values (for areas where trend is NaN)
        trend_filled = trend.fillna(df)

        # Remove both trend and seasonal components for the current period
        stationary_series -= seasonal  # Subtract seasonal component
        stationary_series -= trend_filled  # Subtract trend component

    return stationary_series


# ______________________________________________________________________________________________
# This function allows to make a time series stationary whenever it is not. It receive as an 
# input the dataserie itself, apply the difference at the first order if no seasonality period
# is gave or it applies seasonal differencing. It returns as an output the differenced timeseries,
# unless some error ocurred, then it returns None.

def make_stationary_diff(df, seasonality_period=[]):
    try:
        # Compute the baseline of the original series
        baseline = df.median()
        # Check if the input dataframe is empty
        if df.empty:
            raise ValueError("The input time series is empty.")

        # First-order differencing if no seasonality period is provided
        if not seasonality_period:  # No seasonality
            df_diff = df.diff().dropna()
        
        else:
            # Apply seasonal differencing for each period in the seasonality_period list
            for period in seasonality_period:
                if isinstance(period, (int, float)) and period > 0:
                    # Perform seasonal differencing
                    df_diff = df.diff(int(period)).dropna()
                else:
                    raise ValueError(f"Invalid seasonality period: {period}. It should be a positive integer or float.")
        
        # Add the baseline back to the differenced series
        df_diff += baseline
        return df_diff
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# ______________________________________________________________________________________________
# This function allows to rest the trend from a given time series.  It receives as an input the 
# dataserie itself and the computed decompositions (trend, seasonality and residual), allowing 
# multiple seasonality analysis. The function rests the trends and returns a single detrended 
# time serie.

def rest_trend(df, decompositions):
    # Initialize the detrended series with the original data
    detrended_series= df.copy()

    # Subtract seasonal and trend components from the original data for each seasonality
    for decomposition in decompositions:
        trend = decomposition[0]

        # Fill NaN values in the trend with the original values (for areas where trend is NaN)
        trend_filled = trend.fillna(df)

        # Remove trend component for the current period
        detrended_series -= trend_filled  # Subtract trend component

    return detrended_series

# ______________________________________________________________________________________________
# This function allows to rest the trend from a given time series.  It receives as an input the 
# dataserie itself and the computed decompositions (trend, seasonality and residual), allowing 
# multiple seasonality analysis. The function rests the seasons and returns a single deseasoned 
# time serie.

def rest_seasonality(df, decompositions):
    # Initialize the deseasoned series with the original data
    deseasoned_series = df.copy()

    # Subtract seasonal and trend components from the original data for each seasonality
    for decomposition in decompositions:
        seasonal = decomposition[1]

        # Remove both trend and seasonal components for the current period
        deseasoned_series -= seasonal  # Subtract seasonal component

    return deseasoned_series

# ______________________________________________________________________________________________
# This function allows to rest the trend from a given time series.  It receives as an input the 
# dataserie itself and the computed decompositions (trend, seasonality and residual), allowing 
# multiple seasonality analysis. The function rests the seasons and returns a single deseasoned 
# time serie.


def get_residuals(df, decompositions):
    # Ensure decompositions is not empty
    if not decompositions or len(decompositions) == 0:
        raise ValueError("Decompositions data is missing or empty.")
    
    # Start with the residual from the first decomposition
    residual_series = decompositions[0][2]  # Assuming [2] is the residual component of the decomposition
    
    # Loop through the remaining decompositions and sum the residuals
    for decomposition in decompositions[1:]:
        residual_series += decomposition[2]  # Add residual from each decomposition
    
    return residual_series

'''
# ______________________________________________________________________________________________
# This function allows to encode cyclical feature based on the hours of a day, in order to
# highlight daily seasonalities. It receive as an input the time serie, adds the column of
# encoded hour by the use of sine and cosine and return the enriched time serie.

def add_cyclic_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    return df'''


''''
________________________________________________________________________________________________________
FUNCTIONS FOR FORECASTING ALGORITHM
________________________________________________________________________________________________________
'''

''' In this code we stored the functions that were used in the forecasting section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''


def create_sequences(data, tau):
    # Function to create sequences in correct format for the TDNN
    # INPUT:
    # - data: the time series
    # - tau: the length of the sliding window in input to the TDNN
    # OUTPUT:
    # - sequences: sequence in format: (num_sequences, tau)

    num_sequences = len(data) - tau + 1  # Number of sequences
    sequences = np.zeros((num_sequences, tau))  # Initialize matrix
    for i in range(num_sequences):
        sequences[i] = data[i:i+tau]
    return sequences


def split_data(x_data, y_data, train_size=0.8, val_size=0.1, test_size=0.1):
    # Function to split data into training, validation, and test sets
    # INPUT:
    # - x_data: input data set
    # - y_data: target data set
    # train_size = 0.8, val_size = 0.1 and test_size = 0.1: splitting points
    # OUTPUT:
    # - x_train, x_val, x_test, y_train, y_val, y_test: splitted sets
    assert train_size + val_size + test_size == 1, "The splits should sum to 1"

    # Split into training, validation and test sets
    train_val_size = int(len(x_data) * train_size)  # 80% for training + validation
    x_train_val, x_test = x_data[:train_val_size], x_data[train_val_size:]
    y_train_val, y_test = y_data[:train_val_size], y_data[train_val_size:]

    # Further split the train_val set into training and validation sets
    train_size = int(len(x_train_val) * (train_size / (train_size + val_size)))  # 80% of 80% for training
    x_train, x_val = x_train_val[:train_size], x_train_val[train_size:]
    y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_TDNN(hidden_units, lr):
    # Function that creates the TDNN model
    # The model comprises:
    # - 3 hidden Dense layers of hidden_units neuron with relu as activation function
    # - an output layer
    # The loss is calculated with MSE (Mean Square Error) and the optimizer is Adam
    # INPUT:
    # - hidden_units: number of neurons in the layer
    # - lr: learning rate
    # OUTPUT:
    # - model: the model
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu'))
    #model.add(Dropout(0.2))
    # model.batchnormalization ??
    model.add(Dense(hidden_units, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr))
    return model


def training_TDNN(TDNN_model, x_train, y_train, x_val, y_val, epochs):
    # Function to train the TDNN model
    # INPUT
    # - TDNN_model: the TDNN model
    # - x_train, y_train, x_val, y_val: input and target training and validation sets
    # - epochs: number of epochs
    # OUTPUT:
    # - loss_validation: the loss of the trained model that we want to minimize
    history = TDNN_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), verbose=0)
    print(history.history.keys())
    loss_training = history.history['loss']
    print('Loss training: ', loss_training[-1])
    loss_validation = history.history['val_loss']
    print('Loss validation: ', loss_validation[-1])
    return loss_validation[-1]


def objective_TDNN(trial, time_series):
    # Function that uses Optuna for hyperparameters optimization
    # HYPERPARAMETERS:
    # - tau: length of input sliding window to TDNN
    # - epochs: number of epochs to train TDNN
    # - lr: learning rate
    # - hidden_units: number of neurons in TDNN layers
    # INPUT:
    # - trial: number of trials of the study to search for hyperparameters
    # - time series: time series we want to train
    # OUTPUT:
    # - val_loss: loss of the validation that we want to minimize

    # Set hyperparameters ranges
    tau = trial.suggest_categorical('tau', [8 ,15 , 22])
    epochs = trial.suggest_int('epochs', 50, 150, step=10)
    lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
    hidden_units = trial.suggest_int('hidden_units', 50, 250)
    TDNN_model = create_TDNN(hidden_units, lr)

    # Create sequences for the model
    sequences = create_sequences(time_series, tau)
    x_data = sequences[:, :-1]  # All but the last value as features
    y_data = time_series[tau-1:]  # The corresponding targets

    #print(x_data.shape)
    #print(y_data.shape)

    # Split data into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_data, y_data)

    # Compute mean and std from training data
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    # Normalize training and test data with mean and variance of the training
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    # Reshape the input data to (1, num_sequences, tau)
    x_train = np.expand_dims(x_train, axis=0)  # Shape (1, num_sequences, tau)
    x_val = np.expand_dims(x_val, axis=0)  # Shape (1, num_sequences, tau)

    # Reshape target data to (1, num_sequences)
    y_train = np.expand_dims(y_train, axis=0)  # Shape (1, num_sequences)
    y_val = np.expand_dims(y_val, axis=0)  # Shape (1, num_sequences)
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_val.shape)
    #print(y_val.shape)

    # Train the model and return the validation loss
    val_loss = training_TDNN(TDNN_model, x_train, y_train, x_val, y_val, epochs)
    return val_loss


def tdnn_forecasting_training(series, n_trials=10):
    # Function that creates a study for hyperparameter optimization
    # and validates the TDNN using the best hyperparameters found
    # INPUT:
    # - time series: time series we want to train and find best model and hyperparameters
    #                this series has a column 'time'
    #                and a column with one of ['min', 'max', 'sum', 'avg']
    # - n_trials: number of trials for hyperparameter search
    # OUTPUT:
    # - best_model_TDNN: model of TDNN with best hyperparameters
    # - best_params: dictionary comprising best hyperparameters ['tau', 'lr', 'epochs', 'hidden_units']
    # - stats: array comprising [x_mean, x_std, y_mean, y_std] which are needed for proper normalization

    # Extract only column associated to one of ['min', 'max', 'sum', 'avg']
    time_series = series.iloc[:, 1]

    # Create study and save best params
    TDNN_study = optuna.create_study(direction='minimize')
    TDNN_study.optimize(lambda trial: objective_TDNN(trial, time_series), n_trials=n_trials)
    best_params = TDNN_study.best_params
    #print('Best Hyperparameters:', best_params)
    tau = best_params['tau']
    epochs = best_params['epochs']
    hidden_units = best_params['hidden_units']
    lr = best_params['lr']

    # Create model with best hyperparameters
    best_model_TDNN = create_TDNN(hidden_units, lr)

    # Split time_series into input and target
    sequences = create_sequences(time_series, tau)
    x_data = sequences[:, :-1]  # All but the last value as features
    y_data = time_series[tau-1:]  # The corresponding targets

    # Split data into training, validation, and test sets
    x_training, x_val, x_test, y_training, y_val, y_test = split_data(x_data, y_data)

    # Compute mean and std from training data
    x_mean = np.mean(x_training)
    x_std = np.std(x_training)
    y_mean = np.mean(y_training)
    y_std = np.std(y_training)
    stats = np.array([x_mean, x_std, y_mean, y_std])

    # Normalize training and test data with training stats
    x_training = (x_training - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_training = (y_training - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Reshape input data to (1, num_sequences, tau)
    x_training = np.expand_dims(x_training, axis=0)  # Shape (1, num_sequences, tau)
    x_test = np.expand_dims(x_test, axis=0)  # Shape (1, num_sequences, tau)

    # Reshape target data to (1, num_sequences)
    y_training = np.expand_dims(y_training, axis=0)  # Shape (1, num_sequences)
    y_test = np.expand_dims(y_test, axis=0)  # Shape (1, num_sequences)

    # Train the model
    history = best_model_TDNN.fit(x_training, y_training, epochs=epochs, verbose=0)
    # print(history.history['loss'][-1])

    # Predict on training and test data
    y_pred_training = best_model_TDNN.predict(x_training).reshape(-1)
    y_pred_test = best_model_TDNN.predict(x_test).reshape(-1)

    # calculate MSE
    TDNN_test_MSE = best_model_TDNN.evaluate(x_test, y_test)
    print('Test MSE: ', TDNN_test_MSE)

    # Denormalize predictions and targets for plotting
    y_pred_training = y_pred_training * y_std + y_mean
    y_pred_test = y_pred_test * y_std + y_mean
    y_training = y_training * y_std + y_mean
    y_test = y_test * y_std + y_mean

     # Get time indexes for training and test data
    time_indexes_training = series.iloc[:len(y_training.reshape(-1)), 0]

    # Calculate the starting index for the test data in the original time series
    test_start_index = len(series) - len(y_test.reshape(-1))
    time_indexes_test = series.iloc[test_start_index:, 0]  # Get time indexes for test data

    '''# Plot the results
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(time_indexes_training, y_training.reshape(-1), label='Target')
    plt.plot(time_indexes_training, y_pred_training, label='Predicted')
    plt.title('Predicted Training vs. Target Training')
    plt.legend()
    plt.xlabel('Time')
    plt.subplot(2, 1, 2)
    plt.plot(time_indexes_test, y_test.reshape(-1), label='Target')
    plt.plot(time_indexes_test, y_pred_test, label='Predicted')
    plt.title('Predicted Test vs. Target Test')
    plt.xlabel('Time')
    plt.legend()
    plt.show()'''


    return [best_model_TDNN, best_params, stats]



def tdnn_forecasting_prediction(model, tau, time_series, stats, timestamp_init = None, timestamp_end = None):
    # Function that uses the trained model to predict num_predictions in the future
    # INPUT:
    # - model: TDNN best model after training
    # - tau: length of input sliding window which can be retrieved from best_params['tau']
    # - time_series: this series has a column 'time'
    #                and a column with one of ['min', 'max', 'sum', 'avg']:
    # - timestamp_init: begin date of prediction in days format
    #                   default value = None
    # - timestamp_end: end date of prediction in days format
    #                  default value = None
    # - stats: list with statistics to normalize the time series
    # OUTPUT:
    # - predictions_df: a dataframe with first column named 'time' with prediction_timestamps
    #                   and second column with name in ['min', 'max', 'sum', 'avg'] with predicted values
    x_mean, x_std, y_mean, y_std = stats
    time_series['time'] = pd.to_datetime(time_series['time'])
    series = time_series.iloc[:,1]
    column_name = time_series.columns[1]
    initial_window = np.array(series[len(series)-tau+1:])  # Use the last sequence as the initial window
    predictions = []
    current_window = (initial_window - x_mean) / x_std  # Normalize the input window

    # Get last time_stamp and add one as start index of prediction
    time_indexes = time_series.iloc[:, 0]

    # Convert time_indexes to timezone-naive datetime objects if they are timezone-aware
    time_indexes = time_indexes.dt.tz_localize(None)

    # Get the last timestamp from time_indexes if timestamp_init is not given
    if timestamp_init == None:
        timestamp_init = time_indexes.iloc[-1] + pd.DateOffset(days=1)
    else:
        timestamp_init = pd.to_datetime(timestamp_init)

    # Predict for next 7 days if timestamp_end is not given
    if timestamp_end == None:
        timestamp_end = timestamp_init + pd.DateOffset(days=7)
    else:
        timestamp_end = pd.to_datetime(timestamp_end)

    # Calculate num_prediction as difference in days
    num_predictions = int((timestamp_end - timestamp_init).days) + 1

    # Create prediction_timestamps as a Pandas DatetimeIndex
    prediction_timestamps = pd.date_range(start=timestamp_init, periods=num_predictions, freq='D')

    if time_series.iloc[:,0].dt.tz is not None:  # If original data has timezone, apply it
        if prediction_timestamps.tz is None:
            prediction_timestamps = prediction_timestamps.tz_localize(time_series.iloc[:,0].dt.tz)
        else:
            prediction_timestamps = prediction_timestamps.tz_convert(time_series.iloc[:,0].dt.tz)

    for _ in range(num_predictions):
        # Predict the next value
        # Reshape the input to match the model's expected input shape (1, 1, tau-1)
        next_value_norm = model.predict(current_window.reshape(1, 1, -1))
        next_value_norm = next_value_norm.reshape(-1)  # Convert to 1D array

        next_value = next_value_norm * y_std + y_mean  # Denormalize the prediction
        predictions.append(next_value)

        # Update the current window for the next prediction
        current_window = np.append(current_window[1:], (next_value - x_mean) / x_std)

    predictions_df = pd.DataFrame({'time': prediction_timestamps, column_name: predictions})
    return predictions_df