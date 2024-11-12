''' In this code we stored the functions that were used in the 
preprocessing pipeline, including a brief description of their
inputs, outputs and functioning'''


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

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
# This function performs the decomposition  of the time serie into its trend, season
# and residual components (whenever it's possible to implement the analysis). It returns 
# the decomposed time series in a list, of form [trend, seasonal, residual], unless
# there isn't sufficient data or if some error occurs, in that case it returns None.

def seasonal_additive_decomposition(dataframe, period_observation):
    # Check if the filtered DataFrame has enough data for the decomposition
    if dataframe.empty:
        #print(f"No data found for the time serie. Skipping decomposition.")
        return None  

    # Drop NaN values and check if there are enough observations
    series = dataframe.dropna()

    if len(series) < 2:  # Check if the series has at least 2 observations
        print(f"Not enough data. Skipping decomposition.")
        return None  

    # Ensure the period is correctly set (you may need to adjust this depending on your data's frequency)
    period = len(period_observation)  # You may need to adapt this if months doesn't represent the full seasonality cycle

    if len(series) < 2 * period:  # Ensure enough data points for at least two full cycles
        print(f"Not enough data for two full cycles. Skipping decomposition.")
        return None  

    # Classical decomposition (additive model)
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)

        # Plot the decomposition
        #plt.figure(figsize=(10, 8))
        #decomposition.plot()
        #plt.suptitle(f'Classical Decomposition of Time Series, fontsize=16)
        #plt.show()

        # Access the individual components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        return [trend, seasonal, residual]

    except ValueError as e:
        #print(f"Error during decomposition: {e}")
        return None  

# ______________________________________________________________________________________________
# This function allows to encode cyclical feature based on the hours of a day, in order to
# highlight daily seasonalities. It receive as an input the time serie, adds the column of
# encoded hour by the use of sine and cosine and return the enriched time serie.

def add_cyclic_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    return df

# ______________________________________________________________________________________________
# This function 


def make_stationary(df, kwargs):
    # Apply differencing or seasonal adjustments to make the data stationary
    differencing_order = kwargs.get('differencing_order', 1)  # Default: First-order differencing
    df['value_diff'] = df['value'].diff(periods=differencing_order)
    return df

# ______________________________________________________________________________________________
# This function takes in input the data point that we are receiving and checks its reliability in terms of format. 
# In output it will return nothing if the data point is too incomplete to correctly identify and use it, or the reconstructed data point with nans whenever some value is missing.

from collections import OrderedDict
from dateutil import parser
from datetime import timezone

def validate_data_format(x):
    # Check if all essential columns are present: if the first four fields are missing then
    # the identity of the data point is unknown, thus, it needs to be deleted. 
    # Otherwise, (one of the remaining 4 fields in missing), set the missing column as missing. 
    # If all the aggregates are missing the data point is discarded as well.
    expected_columns = ['time', 'asset_id', 'name', 'kpi', 'sum', 'avg', 'min', 'max']
    missing_columns = [col for col in expected_columns if col not in list(x.keys())]
    essential_missing_columns=list(np.intersect1d(expected_columns[:4], list(x.keys())))
    optional_missing_columns=list(np.intersect1d(expected_columns[4:], list(x.keys())))
    if any(item in expected_columns[:4] for item in missing_columns) or any(pd.isna(item) for item in [x.get(key) for key in essential_missing_columns]):
        print('Data point misses essential fields. Discarded.')
        return None
    elif all(pd.isna(item) for item in [x.get(key) for key in optional_missing_columns]):
        print('Data point containing only nan values. Discarded')
        return None
    else:
        for col in missing_columns:
            x[col] = np.nan
    x = dict(OrderedDict((key, x[key]) for key in expected_columns))

    # Try to set the format of the 'time' field into the most used one (ISO 8601 format in UTC)
    try:
        date_obj = parser.parse(x['time'])
        x['time'] = date_obj.replace(tzinfo=timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    except Exception as e:
        print("Invalid time format. Discarded data point.")
        return None
    
    # Check if the aggregations (min, max, sum, avg) sarisfy the basic logic rule min<=avg<=max<=sum
    dp_aggregations=[z for z in [x['min'], x['avg'], x['max'], x['sum']] if not np.isnan(z)]
    if all(dp_aggregations[i] <= dp_aggregations[i+1] for i in range(len(dp_aggregations)-1)):
        pass
    else:
        for col in expected_columns[4:]:
            x[col]=np.nan
            
    return x

# ______________________________________________________________________________________________
# This function is an exemplified version of a set of checks regarding the range that the data point value can assume according to the specific physical quantity that it represents.

def check_range(x):
    # The check of range depends on the nature of the data point. For sensor data, the range requires 
    # knowledge of the machine characteristics and functioning.
    # In a prototype framework we assume to receive sensor data about temperature in K ... 
    max_temp=500
    if x['kpi']=='temperature':
        if (x['min'] >= 0) and (not pd.isna(x['min'])) and (x['max'] <= max_temp) and (not pd.isna(pd.isna(x['max']))):
            pass
        else: #the data point doesn't satisfy the range according to the physical meaning of the quantity that has been measured.
            for agg in ['sum', 'avg', 'min', 'max']:
                x[agg]=np.nan
    return x


# ______________________________________________________________________________________________
# This function is the one that allow the imputation of missing values.
# As a first version we use the average of the previous 14 points in the timeseries as 
# an imputation method even if suboptimal. Future modifications will include more sophiasticated methods
# able to produce a better estimation of the missing value, based on the recent history of the timeseries (SARIMA)
def impute(x, buffer_size):
    buffer=[]
    for key, value in list(x.items())[-4:]:
        if np.isnan(value):
            values = [dp[key] for dp in buffer if not np.isnan(dp[key])]
            if values:  
                x[key] = np.mean(values)

    buffer.append(x)

    if len(buffer) > buffer_size:
        buffer.pop(0) 
    return x
