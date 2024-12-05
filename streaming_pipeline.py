from dataprocessing_functions  import cleaning_pipeline
from dataprocessing_functions  import ad_predict, ad_train
from dataprocessing_functions  import ADWIN_drift
from dataprocessing_functions import tdnn_forecasting_training, tdnn_forecasting_prediction
from dataprocessing_functions import  feature_engineering_pipeline
from dataprocessing_functions import  get_model_ad, update_model_forecast, update_model_ad, identity
from connections_functions import get_datapoint, get_historical_data, send_alert, store_datapoint



#initializing the anomaly detector
counter = 0
while True: #loops continuosly

    #first we call get_datapoint and we wait for a new input to arrive
    new_datapoint = get_datapoint(counter) ## CONNECTION WITH API
    counter += 1
    #once the new data point is aquired we clean it
    cleaned_datapoint = cleaning_pipeline(new_datapoint)

    #we now check if some drift has been detected
    drift_flag=ADWIN_drift(cleaned_datapoint)

    #we call the database to extract historical data
    historical_data = get_historical_data(cleaned_datapoint['name'], cleaned_datapoint['asset_id'], cleaned_datapoint['kpi'], cleaned_datapoint['operation'], -1 , -1) ## CONNECTION WITH API
    

    if drift_flag==True:

        #retrain anomaly detection model
        model=ad_train(historical_data) #Here we should put the get_historical_data()
        update_model_ad(cleaned_datapoint, model)

        #retrain forecasting algorithm model
        features = ['sum', 'avg','min', 'max', 'var']
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
        
        send_alert(anomaly_identity, 'Anomaly')
    
    store_datapoint(cleaned_datapoint) ## CONNECTION WITH API