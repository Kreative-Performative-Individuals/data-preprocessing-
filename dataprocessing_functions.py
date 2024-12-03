''' In this code we stored the functions that were used in the data processing pipeline,
including a brief description of their inputs, outputs and functioning'''


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import json
from collections import OrderedDict, deque
from dateutil import parser
from datetime import timezone, datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from river import drift
import optuna


''''
________________________________________________________________________________________________________
FUNCTIONS FOR INFO MANAGER
________________________________________________________________________________________________________
'''
''' In this code we stored the functions that were used in the info manager section of the
preprocessing pipeline, including a brief description of their inputs, outputs and functioning'''

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

def ADWIN_drift(x, delta=0.002, clock=10):

    for f in features:

        # Check if the column exists in the DataFrame
        batch = get_batch(x, f)
        batch1 = batch[:-1]
        batch2 = batch[1:]
        adwin = drift.ADWIN(delta=0.05, clock=10)
        flag1=False
        flag2=False

        for i, value in enumerate(batch1):
            adwin.update(value)
            if adwin.drift_detected and i>b_length-3:
                flag1=True
        for i, value in enumerate(batch2):
            adwin.update(value)
            if adwin.drift_detected and i>b_length-3:
                flag2=True
        if flag1==False and flag2==True:
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


class AnomalyDetector:
    def __init__(self):
        self.features=['sum', 'avg', 'min', 'max', 'var']
        
    def train(self, hist_ts):
        train_set=pd.DataFrame(hist_ts)[self.features]
        s=[]
        cc=np.arange(0.01, 0.5, 0.01)
        for c in cc:
            model = IsolationForest(n_estimators=200, contamination=c)
            an_pred=model.fit_predict(train_set)
            s.append(silhouette_score(train_set, an_pred))
        optimal_c=cc[np.argmax(s)]
        #print(optimal_c)
        model = IsolationForest(n_estimators=200, contamination=optimal_c)
        model.fit_predict(train_set)
        return model 
    
    def predict(self, x, model):
        dp=pd.DataFrame(x[self.features]).T
        anomaly=model.predict(dp)
        if anomaly==-1:
            anomaly='Anomaly'
        else:
            anomaly='Normal'
        return 
''''
________________________________________________________________________________________________________
FUNCTIONS FOR FEATURES ENGINEERING
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
    features = ['sum', 'avg','min', 'max', 'var']
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

    return dataframe
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
    return highest_acf_lag


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
        period = len(dataframe)

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



'''# ______________________________________________________________________________________________
Function to implement if we want to allow decomposition having more than one seasonality

def seasonal_stl_decomposition(dataframe, periods):
    """
    This function performs seasonal decomposition with STL (Seasonal and Trend decomposition using LOESS)
    for detecting multiple seasonalities. It allows for decomposition of the series into trend, seasonal,
    and residual components for each specified period of seasonality.

    Args:
    - dataframe (pandas.Series): The time series data.
    - periods (list): A list of seasonal periods to consider for decomposition (e.g., [7, 365] for weekly and yearly seasonality).
    
    Returns:
    - list of dicts: Each dict contains 'trend', 'seasonal', 'residual' components for each seasonality.
    """
    # Check if the DataFrame has enough data
    if dataframe.empty:
        print(f"No data found for the time series. Skipping decomposition.")
        return None

    # Drop NaN values
    series = dataframe.dropna()

    if len(series) < 2:  # Check if the series has at least 2 observations
        print(f"Not enough data. Skipping decomposition.")
        return None

    # Store decomposition results for each period
    decompositions = []

    # Perform STL decomposition for each period in the list
    for period in periods:
        if len(series) < 2 * period:  # Ensure enough data for two full cycles for each seasonality
            print(f"Not enough data for two full cycles of period {period}. Skipping decomposition.")
            continue
        
        try:
            # Apply STL decomposition
            decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=period)

            # Store the decomposition results
            decomposition_result = {
                'period': period,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            decompositions.append(decomposition_result)

        except ValueError as e:
            print(f"Error during decomposition with period {period}: {e}")
            continue

    # Return the decomposition results
    if decompositions:
        return decompositions
    else:
        print("No decompositions were successful.")
        return None'''

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
    model.add(Dense(hidden_units, activation='relu'))
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
    tau = trial.suggest_categorical('tau', [7 ,14 , 21])
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


def tdnn_forecasting_training(time_series, n_trials=10):
    # Function that creates a study for hyperparameter optimization
    # and validates the TDNN using the best hyperparameters found
    # INPUT:
    # - time series: time series we want to train and find best model and hyperparameters
    # - n_trials: number of trials for hyperparameter search
    # OUTPUT:
    # - best_model_TDNN: model of TDNN with best hyperparameters
    # - best_params: dictionary comprising best hyperparameters ['tau', 'lr', 'epochs', 'hidden_units']
    # - stats: array comprising [x_mean, x_std, y_mean, y_std] which are needed for proper normalization

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
    # print('Test MSE: ', TDNN_test_MSE)

    # Denormalize predictions and targets for plotting
    #y_pred_training = y_pred_training * y_std + y_mean
    #y_pred_test = y_pred_test * y_std + y_mean
    #y_training = y_training * y_std + y_mean
    #y_test = y_test * y_std + y_mean

    # Plot the results
    #plt.figure(figsize=(12, 10))
    #plt.subplot(2, 1, 1)
    #plt.plot(y_training.reshape(-1), label='Target')
    #plt.plot(y_pred_training, label='Predicted')
    #plt.title('Predicted Training vs. Target Training')
    #plt.legend()
    #plt.xlabel('Time')
    #plt.subplot(2, 1, 2)
    #plt.plot(y_test.reshape(-1), label='Target')
    #plt.plot(y_pred_test, label='Predicted')
    #plt.title('Predicted Test vs. Target Test')
    #plt.xlabel('Time')
    #plt.legend()
    #plt.show()

    return [best_model_TDNN, best_params, stats]



def tdnn_forecasting_prediction(model, tau, time_series, num_predictions, stats):
    # Function that uses the trained model to predict num_predictions in the future
    # INPUT:
    # - model: TDNN best model after training
    # - tau: length of input sliding window which can be retrieved from best_params['tau']
    # - num_predictions: number of future step to predict
    # - stats: list with statistics to normalize the time series

    x_mean, x_std, y_mean, y_std = stats
    sequences = create_sequences(time_series, tau)
    initial_window = sequences[-1, :-1]  # Use the last sequence as the initial window
    predictions = []
    current_window = (initial_window - x_mean) / x_std  # Normalize the input window

    for _ in range(num_predictions):
        # Predict the next value
        # Reshape the input to match the model's expected input shape (1, 1, tau-1)
        next_value_norm = model.predict(current_window.reshape(1, 1, -1))[0, 0]
        next_value = next_value_norm * y_std + y_mean  # Denormalize the prediction
        predictions.append(next_value)

        # Update the current window for the next prediction
        current_window = np.append(current_window[1:], (next_value - x_mean) / x_std)

    return predictions
