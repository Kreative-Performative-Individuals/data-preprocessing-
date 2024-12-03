from machine_learning_functions import tdnn_forecasting_prediction
from feature_engineering_functions import feature_engineering_pipeline
from infoManager import get_model_forecast
from connections_functions import get_historical_data


def get_request(machine_name, asset_id, kpi, operation, timestap_start, timestamp_end, transformation, forecasting):
    

    transformation_config = {
        'make_stationary': False,  # Default: False
        'detrend': False,          # Default: False
        'deseasonalize': False,    # Default: False
        'get_residuals': False,    # Default: False
        'scaler': False             # Default: False
    }

    if forecasting:
        transformation_config['make_stationary'] = True
        transformation_config['scaler'] = True
        
        historical_data = get_historical_data(machine_name, asset_id, kpi, operation, -1, -1) ## CONNECTION WITH API

        transformed_data = feature_engineering_pipeline(historical_data, transformation_config)

        forecasting_model_info = get_model_forecast(historical_data[-1])

        forecasting_model = forecasting_model_info[0]
        forecasting_params = forecasting_model_info[1]
        forecasting_stats = forecasting_model_info[2]
        predictions = tdnn_forecasting_prediction(forecasting_model, forecasting_params['tau'], transformed_data, timestap_start, timestamp_end, forecasting_stats)

        json_predictions = predictions.to_json(orient='records')

        return json_predictions
    
    else:
        if transformation == 'S':
            transformation_config['detrend'] = True
        elif transformation == 'T':
            transformation_config['deseasonalize'] = True
            
        historical_data = get_historical_data(machine_name, asset_id, kpi, operation, timestap_start, timestamp_end) ## CONNECTION WITH API

        transformed_data = feature_engineering_pipeline(historical_data, transformation_config)

        json_transformed_data = transformed_data.to_json(orient='records')

        return json_transformed_data

    