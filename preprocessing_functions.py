''' In this code we stored the functions that were used in the 
preprocessing pipeline, including a brief description of their
inputs, outputs and functioning'''


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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
    

def seasonal_additive_decomposition(dataframe):
    # Check if the filtered DataFrame has enough data for the decomposition
    if filtered_df.empty:
        print(f"No data found for machine '{machine}', KPI '{kpi}', value '{value}'. Skipping decomposition.")
        continue  # Skip this iteration

    # Drop NaN values and check if there are enough observations
    series = filtered_df[value].dropna()

    if len(series) < 2:  # Check if the series has at least 2 observations
        print(f"Not enough data for machine '{machine}', KPI '{kpi}', value '{value}'. Skipping decomposition.")
        continue  # Skip this iteration

    # Ensure the period is correctly set (you may need to adjust this depending on your data's frequency)
    period = len(months)  # You may need to adapt this if months doesn't represent the full seasonality cycle

    if len(series) < 2 * period:  # Ensure enough data points for at least two full cycles
        print(f"Not enough data for two full cycles for machine '{machine}', KPI '{kpi}', value '{value}'. Skipping decomposition.")
        continue  # Skip this iteration

    # Classical decomposition (additive model)
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)

        # Plot the decomposition
        plt.figure(figsize=(10, 8))
        decomposition.plot()
        plt.suptitle(f'Classical Decomposition of Time Series for {machine} - {kpi} - {value}', fontsize=16)
        plt.show()

        # Access the individual components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

    except ValueError as e:
        print(f"Error during decomposition for machine '{machine}', KPI '{kpi}', value '{value}': {e}")
        continue  # Skip this iteration if an error occurs during decomposition

