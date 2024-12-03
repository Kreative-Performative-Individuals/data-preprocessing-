from dataprocessing_functions  import cleaning_pipeline
from dataprocessing_functions  import AnomalyDetector
from dataprocessing_functions  import ADWIN_drift
from dataprocessing_functions import tdnn_forecasting_training, tdnn_forecasting_prediction
from dataprocessing_functions import  feature_engineering_pipeline
from dataprocessing_functions import  get_model_ad, update_model_forecast, update_model_ad, identity
from connections_functions import get_datapoint, get_historical_data, send_alert, store_datapoint



#initializing the anomaly detector
ad=AnomalyDetector()

while True: #loops continuosly

    #first we call get_datapoint and we wait for a new input to arrive
    new_datapoint = get_datapoint(10) ## CONNECTION WITH API

    #once the new data point is aquired we clean it
    cleaned_datapoint = cleaning_pipeline(new_datapoint)

    #we now check if some drift has been detected
    drift_flag=ADWIN_drift(cleaned_datapoint)

    #we call the database to extract historical data
    historical_data = get_historical_data(cleaned_datapoint['name'], cleaned_datapoint['asset_id'], cleaned_datapoint['kpi'], cleaned_datapoint['operation'], None , None) ## CONNECTION WITH API
    

    if drift_flag==True:

        #retrain anomaly detection model
        model=ad.train(historical_data) #Here we should put the get_historical_data()
        update_model_ad(cleaned_datapoint, model)

        #retrain forecasting algorithm model
        model_info = tdnn_forecasting_training(historical_data)  #contains [best_model_TDNN, best_params, stats] #Here we should put the get_historical_data()
        update_model_forecast(cleaned_datapoint, model_info)
        
    #Anomalies detection branch 
    # get de model    
    ad_model= get_model_ad(cleaned_datapoint)
    #predict class
    cleaned_datapoint['anomaly']=ad.predict(cleaned_datapoint , ad_model)

    if cleaned_datapoint['anomaly']=="Anomaly":
        anomaly_identity = {key: cleaned_datapoint[key] for key in identity if key in cleaned_datapoint}
        
        _ = send_alert(anomaly_identity)
    
    _ = store_datapoint(cleaned_datapoint) ## CONNECTION WITH API