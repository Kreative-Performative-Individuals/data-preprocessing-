from dataprocessing_functions import tdnn_forecasting_prediction, feature_engineering_pipeline, get_model_forecast
from connection_functions import get_historical_data
import pandas as pd


def get_request(
    machine_name,
    asset_id,
    kpi,
    operation,
    timestap_start,
    timestamp_end,
    transformation,
    forecasting,
):

    transformation_config = {
        "make_stationary": False,  # Default: False
        "detrend": False,  # Default: False
        "deseasonalize": False,  # Default: False
        "get_residuals": False,  # Default: False
        "scaler": False,  # Default: False
    }

    if forecasting:
        # transformation_config['make_stationary'] = True
        # transformation_config['scaler'] = True

        historical_data = get_historical_data(
            machine_name, asset_id, kpi, operation, -1, -1
        )  ## CONNECTION WITH API

        # transformed_data = feature_engineering_pipeline(historical_data, transformation_config)

        forecasting_model_info = get_model_forecast(historical_data.iloc[-1])

        data_predictions = []
        # for feature_name, feature_model_info in models.items():
        for feature_name, feature_model_info in forecasting_model_info.items():
            if feature_name in historical_data.columns:
                feature = historical_data[["time", feature_name]]
                if not (
                    feature[feature_name].empty
                    or feature[feature_name].isna().all()
                    or feature[feature_name].isnull().all()
                ):
                    forecasting_model = feature_model_info[0]
                    forecasting_params = feature_model_info[1]
                    forecasting_stats = feature_model_info[2]
                    predictions = tdnn_forecasting_prediction(
                        forecasting_model,
                        forecasting_params["tau"],
                        feature,
                        forecasting_stats,
                        timestap_start,
                        timestamp_end,
                    )
                    data_predictions.append(predictions)

        data_predictions = pd.concat(data_predictions, axis=1)

        # Drop the duplicate 'time' column after concatenation
        data_predictions = data_predictions.loc[
            :, ~data_predictions.columns.duplicated()
        ]

        json_predictions = data_predictions.to_json(orient="records")

        return json_predictions

    else:
        if transformation == "S":
            transformation_config["detrend"] = True
        elif transformation == "T":
            transformation_config["deseasonalize"] = True

        historical_data = get_historical_data(
            machine_name, asset_id, kpi, operation, timestap_start, timestamp_end
        )  ## CONNECTION WITH API

        transformed_data = feature_engineering_pipeline(
            historical_data, transformation_config
        )

        json_transformed_data = transformed_data.to_json(orient="records")

        return json_transformed_data
