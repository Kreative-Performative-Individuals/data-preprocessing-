from dataprocessing_functions  import cleaning_pipeline, ad_predict, ad_train, ADWIN_drift, tdnn_forecasting_training,  get_model_ad, update_model_forecast, update_model_ad, identity, features, feature_engineering_pipeline
from connections_functions import get_datapoint, get_historical_data, send_alert, store_datapoint

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

#initializing the anomaly detector
c = 120
while c<490: #loops continuosly
    #first we call get_datapoint and we wait for a new input to arrive
    new_datapoint = get_datapoint(c) ## CONNECTION WITH API
    print(new_datapoint)
    # new_datapoint={
    # 'time': str(datetime.now()),
    # 'asset_id':  'ast-yhccl1zjue2t',
    # 'name': 'metal_cutting',
    # 'kpi': 'time',
    # 'operation': 'working',
    # 'sum': 1,
    # 'avg': -1,
    # 'min': -1,
    # 'max': -1,
    # 'var': -1}
    #once the new data point is aquired we clean it
    cleaned_datapoint = cleaning_pipeline(new_datapoint)
    print(cleaned_datapoint)

    if cleaned_datapoint:
        #we now check if some drift has been detected
        drift_flag=ADWIN_drift(cleaned_datapoint)

        #we call the database to extract historical data
        historical_data = get_historical_data(cleaned_datapoint['name'], cleaned_datapoint['asset_id'], cleaned_datapoint['kpi'], cleaned_datapoint['operation'], -1 , cleaned_datapoint['time']) ## CONNECTION WITH API
        
        
        if drift_flag==True:

            #retrain anomaly detection model
            model=ad_train(historical_data) #Here we should put the get_historical_data()
            update_model_ad(cleaned_datapoint, model)

            #retrain forecasting algorithm model
            models = {}
            for feature_name in features:
                # Check if the column exists in the DataFrame
                if feature_name in historical_data.columns:
                    feature = historical_data[['time',feature_name]]
                    if not (feature[feature_name].empty or feature[feature_name].isna().all() or feature[feature_name].isnull().all()):
                        model_info = tdnn_forecasting_training(feature)  #contains [best_model_TDNN, best_params, stats] 
                        models[feature_name] = model_info
            update_model_forecast(cleaned_datapoint, models) #a dictionary with models for each feature are stored
            
            
        #Anomalies detection branch 
        # get de model    
        ad_model= get_model_ad(cleaned_datapoint)

        #predict class
        cleaned_datapoint['status'], anomaly_score=ad_predict(cleaned_datapoint , ad_model)

        if cleaned_datapoint['status']=="Anomaly":
            anomaly_identity = {key: cleaned_datapoint[key] for key in identity if key in cleaned_datapoint}
            
            send_alert(anomaly_identity, 'Anomaly', None, anomaly_score)
        
        transformation_config = {
        'make_stationary': False,  # Default: False
        'detrend': False,          # Default: False
        'deseasonalize': True,    # Default: False
        'get_residuals': False,    # Default: False
        'scaler': False             # Default: False
        }
        to_transform_data =  historical_data.copy()
        print(to_transform_data.T)
        transformed_data = feature_engineering_pipeline(to_transform_data, transformation_config)

        # List of features to plot
        features = ['sum', 'avg', 'min', 'max']
        
        # Create a figure and axes for the plots
        fig, axes = plt.subplots(len(features), 1, figsize=(10, 8), sharex=True)
        
        # Iterate over the features and plot them
        for i, feature in enumerate(features):
            ax = axes[i]  # Select the current axis for plotting
            
            # Plot historical data for the feature
            ax.plot(historical_data['time'], historical_data[feature], label=f'Historical {feature}', color='blue', linestyle='-', marker='o')
            
            # Plot transformed data for the feature
            ax.plot(transformed_data['time'], transformed_data[feature], label=f'Transformed {feature}', color='orange', linestyle='--', marker='x')
            
            # Add title and labels
            ax.set_title(f'{feature.capitalize()} Comparison')
            ax.set_ylabel(feature.capitalize())
            ax.legend()
        
        # Set common x-axis label
        plt.xlabel('Time')
        
        # Adjust layout to avoid overlap of subplots
        plt.tight_layout()
        
        # Show the plots
        plt.show()




        store_datapoint(cleaned_datapoint, c) ## CONNECTION WITH API
        c += 1