from dataprocessing_functions import (
    b_length,
    cleaning_pipeline,
    ad_predict,
    ad_train,
    ADWIN_drift,
    tdnn_forecasting_training,
    get_model_ad,
    update_model_forecast,
    update_model_ad,
    identity,
    features,
)
from connections_functions import (
    get_datapoint,
    get_historical_data,
    send_alert,
    store_datapoint,
)

import warnings

warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np

# initializing the anomaly detector
c = 0
from_drift = 0
check_drift = True
while c < 480:  # loops continuosly
    # first we call get_datapoint and we wait for a new input to arrive
    new_datapoint = get_datapoint(c)  ## CONNECTION WITH API
    print(f"\n{new_datapoint['time'][:10]}")
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
    # once the new data point is aquired we clean it
    cleaned_datapoint = cleaning_pipeline(new_datapoint)

    if cleaned_datapoint:
        # we now check if some drift has been detected
        if check_drift:
            drift_flag = ADWIN_drift(cleaned_datapoint)

            # we call the database to extract historical data

            if drift_flag:
                print("Detected DRIFT")
                historical_data = get_historical_data(
                    cleaned_datapoint["name"],
                    cleaned_datapoint["asset_id"],
                    cleaned_datapoint["kpi"],
                    cleaned_datapoint["operation"],
                    -1,
                    cleaned_datapoint["time"],
                )  ## CONNECTION WITH API

                # retrain anomaly detection model
                model = ad_train(
                    historical_data
                )  # Here we should put the get_historical_data()
                update_model_ad(cleaned_datapoint, model)

                # #retrain forecasting algorithm model
                # models = {}
                # for feature_name in features:
                #     # Check if the column exists in the DataFrame
                #     if feature_name in historical_data.columns:
                #         feature = historical_data[['time',feature_name]]
                #         if not (feature[feature_name].empty or feature[feature_name].isna().all() or feature[feature_name].isnull().all()):
                #             model_info = tdnn_forecasting_training(feature)  #contains [best_model_TDNN, best_params, stats]
                #             models[feature_name] = model_info
                # update_model_forecast(cleaned_datapoint, models) #a dictionary with models for each feature are stored
                check_drift = False
        if not check_drift:
            from_drift += 1
            if from_drift > 7:
                from_drift = 0
                check_drift = True
                print("I can check the drift again")
        # Anomalies detection branch
        # get de model
        ad_model = get_model_ad(cleaned_datapoint)

        # predict class
        cleaned_datapoint["status"], anomaly_score = ad_predict(
            cleaned_datapoint, ad_model
        )

        if cleaned_datapoint["status"] == "Anomaly":
            anomaly_identity = {
                key: cleaned_datapoint[key]
                for key in identity
                if key in cleaned_datapoint
            }

            send_alert(anomaly_identity, "Anomaly", None, anomaly_score)

        store_datapoint(cleaned_datapoint, c)  ## CONNECTION WITH API
    c += 1
