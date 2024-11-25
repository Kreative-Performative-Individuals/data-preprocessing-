''' In this code we stored the functions that were used in the feature engineering section of the
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
FUNCTIONS FOR FEATURES ENGINEERING
________________________________________________________________________________________________________
'''

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
        plt.figure(figsize=(10, 8))
        decomposition.plot()
        plt.suptitle(f'Classical Decomposition of Time Series', fontsize=16)
        plt.show()

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



