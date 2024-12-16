import json
import pandas as pd
import src.app.config as config
from datetime import timedelta
from src.app.notification.mail_sender import MailSender
from src.app.real_time.request import KPIStreamingRequest, KPIValidator
import requests


def get_datapoint(i):
    """
    Retrieves a specific datapoint from the dataset 'original_adapted_dataset.json' in \data.
    The stream is assumed to start from 16/09/2024, thus from index 38,400 onwards.

    Arguments:
    - i (int): The index of the desired datapoint within the truncated data stream.

    Returns:
    - datapoint (dict): the new datapoint of the stream.

    Example:
        >>> i = 5
        >>> datapoint = get_datapoint(i)
        >>> print(datapoint)
        {'time': '2024-09-17 00:00:00+00:00', 'asset_id': 'ast-o8xtn5xa8y87', 'name': 'Riveting Machine', 'kpi': 'average_cycle_time', 'operation': 'working',
        'sum': nan, 'avg': 1.8807104020608367, 'min': 1.658396446642954, 'max': 1.9876466725348092, 'var': nan, 'status': nan} # Example output
    """

    with open(config.ORIGINAL_ADAPTED_DATA_PATH, "r") as file:
        stream = json.load(file)

    stream = pd.DataFrame(stream)[38400:]
    datapoint = stream.iloc[i].to_dict()

    return datapoint


def get_next_datapoint(kpi_validator: KPIValidator, file=config.CLEANED_PREDICTED_DATA_PATH):
    """
    This function yields the next datapoint from the dataset

    :param kpi_validator: The request object containing the kpis, machines and operations
    :param file: The file containing the dataset
    :return: The next datapoint
    """
    data = pd.read_json(file)
    data = data[data['kpi'].isin(kpi_validator.kpis)]
    data = data[data['name'].isin(kpi_validator.machines)]
    data = data[data['operation'].isin(kpi_validator.operations)]

    # yield each record
    for index, row in data.iterrows():
        yield row.to_dict()


def get_historical_data(machine_name, asset_id, kpi, operation, timestap_start, timestamp_end):
     # In some manner receives data frame filtered from the database in format dataframe
     #Maybe we can define that if we give timestap_start = None, timestamp_end = None,
     #they have to return us x values in the past starting from the last stored point
    
    
     url_db = "http://localhost:8002/"
     
     params = {
     "start_date": timestap_start,
     "end_date": timestamp_end,
     "kpi_name": kpi,
     "machines ": machine_name,
     "operations": operation,
     "asset_id": asset_id,
     "column_name": ""}

     # Send the GET request
     response = requests.get(url_db + "get_real_time_data", params=params)
     print(response)

     return response


# def get_historical_data_mock(machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end):
#     with open(config.HISTORICAL_DATA_PATH, "r") as file:
#         historical = json.load(file)
#     historical_data = pd.DataFrame(historical)

#     historical_data = historical_data[
#         (historical_data['name'] == machine_name) &
#         (historical_data['asset_id'] == asset_id) &
#         (historical_data['kpi'] == kpi) &
#         (historical_data['operation'] == operation) &
#         (historical_data['status'] != 'Corrupted')].reset_index(drop=True)

#     historical_data['time'] = pd.to_datetime(historical_data['time'])

#     #here we are isolating the specific timeseries that we want by filtering the historical data we have stored.
#     if timestamp_end == -1:
#         timestamp_end = historical_data['time'].iloc[-1]

#     if timestamp_start == -1:
#         timestamp_start = timestamp_end - timedelta(days=100)

#     historical_data = historical_data[
#         (historical_data['time'] >= timestamp_start) &
#         (historical_data['time'] <= timestamp_end)]

#     historical_data['time'] = historical_data['time'].astype(str)

#     return historical_data


def send_alert(identity, type, counter=None,
               probability=None):  #the identity returns the type of Kpi and machine for which the anomaly/nan values
    # have been detected, type is 'Anomaly' or 'Nan', counter (is the number of consecutive days in
    # which we have detected nan) is None if type = 'Anomaly'

    """
    Sends an email alert for detected anomalies or persistent NaN values in a machine's KPI.

    Arguments:
    - identity (dict): Details about the machine and KPI, including:
        - `name`, `asset_id`, `kpi`, `operation` (all required).
        - `explanation`: Required for anomalies.
    - type (str): Type of issue detected ('Anomaly' or 'Nan').
    - counter (int, optional): Consecutive days with NaN values (required if `type == 'Nan'`).
    - probability (int, optional): Anomaly probability in percentage (required if `type == 'Anomaly'`).

    Returns:
    - None
    """
    if type == 'Anomaly':
        object = 'Anomaly alert'
        alert = f"Alert anomaly in machine: '{identity['name']}' - asset: '{identity['asset_id']}' - kpi: '{identity['kpi']}' - operation: '{identity['operation']}'! The probability that this anomaly is correct is {probability}%.\n\n{identity['explanation']}"
    else:
        object = 'Malfunctioning alert'
        alert = f"It has been {counter} days that machine: '{identity['name']}' - asset: '{identity['asset_id']}' returns NaN values in kpi: '{identity['kpi']}' - operation: '{identity['operation']}'. Possible malfunctioning either in the acquisition system or in the machine!"

    config.MAILER.send_mail(object, alert)



def store_datapoint(new_datapoint):

    with open(config.NEW_DATAPOINT_PATH, "w") as json_file:
        json.dump(new_datapoint, json_file, indent=1) 


    url_db = "http://localhost:8002/"
    response = requests.post(url_db + "store_datapoint", params=new_datapoint)



# def store_datapoint(new_datapoint):
#     """
#     Stores a new datapoint in both the current and historical data files.

#     Arguments:
#     - new_datapoint (dict): The new datapoint to be stored.

#     Returns:
#     - None
#     """

#     with open(config.NEW_DATAPOINT_PATH, "w") as json_file:
#         json.dump(new_datapoint, json_file, indent=1)

#     with open(config.HISTORICAL_DATA_PATH, "r") as file:
#         historical = json.load(file)
#     historical = pd.DataFrame(historical)
#     historical = pd.concat([historical, pd.DataFrame([new_datapoint])], ignore_index=True)

#     with open(config.NEW_DATAPOINT_PATH, "w") as file:
#         json.dump(historical.to_dict(), file, indent=1)

