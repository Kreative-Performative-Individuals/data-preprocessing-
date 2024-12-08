import json
import pandas as pd
import src.app.config as config
from datetime import timedelta
from src.app.notification.mail_sender import MailSender


def get_datapoint(i):
    with open(config.ORIGINAL_ADAPTED_DATA_PATH, "r") as file:
        stream = json.load(file)

    stream = pd.DataFrame(stream)[38400:]
    datapoint = stream.iloc[i].to_dict()

    return datapoint


def get_next_datapoint(file=config.CLEANED_PREDICTED_DATA_PATH):
    """
    Yields the next datapoint from the json file.

    :param file: the file path to read from
    :return: the next datapoint
    """
    with open(file, "r") as json_file:
        data = json.load(json_file)
        for i in range(len(data["time"])):
            yield {feature: data[feature][str(i)] for feature in data}


# def get_historical_data(machine_name, asset_id, kpi, operation, timestap_start, timestamp_end):
#     # In some manner receives data frame filtered from the database in format dataframe
#     #Maybe we can define that if we give timestap_start = None, timestamp_end = None,
#     #they have to return us x values in the past starting from the last stored point

#     url_db = "http://localhost:8000/"
#     #Use this URL if you are connecting from the compose.
#     #url_db = "http://db:8000/"
#     params = {
#     "machine_name": machine_name,
#     "asset_id": asset_id,
#     "kpi": kpi,
#     "operation": operation,
#     "timestamp_start": timestap_start,
#     "timestamp_end": timestamp_end}
#     # Send the GET request
#     response = requests.get(url_db + "historical_data", params=params)
#     print(response)

#     return response


def get_historical_data(machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end):
    
    with open(config.HISTORICAL_DATA_PATH, "r") as file:
        historical=json.load(file)
    historical_data=pd.DataFrame(historical)

    historical_data= historical_data[
    (historical_data['name'] == machine_name) & 
    (historical_data['asset_id'] == asset_id) & 
    (historical_data['kpi'] == kpi) & 
    (historical_data['operation'] == operation) & 
    (historical_data['status']!= 'Corrupted')].reset_index(drop=True)

    historical_data['time']=pd.to_datetime(historical_data['time'])

    #here we are isolating the specific timeseries that we want by filtering the historical data we have stored.
    if timestamp_end ==-1:
        timestamp_end=historical_data['time'].iloc[-1]

    if timestamp_start==-1:
        timestamp_start=timestamp_end - timedelta(days=100)       

    historical_data= historical_data[
    (historical_data['time'] >= timestamp_start) & 
    (historical_data['time'] <= timestamp_end)]
    
    historical_data['time']=historical_data['time'].astype(str)

    return historical_data



def send_alert(identity, type, counter=None, probability=None): #the identity returns the type of Kpi and machine for which the anomaly/nan values
                                        # have been detected, type is 'Anomaly' or 'Nan', counter (is the number of consecutive days in
                                        # which we have detected nan) is None if type = 'Anomaly'
    if type == 'Anomaly':
        object='Anomaly alert'
        alert = f"Alert anomaly in machine: '{identity['name']}' - asset: '{identity['asset_id']}' - kpi: '{identity['kpi']}' - operation: '{identity['operation']}'! The probability that this anomaly is correct is {probability}%."
    else:
        object='Malfunctioning alert' 
        alert = f"It has been {counter} days that machine: '{identity['name']}' - asset: '{identity['asset_id']}' returns NaN values in kpi: '{identity['kpi']}' - operation: '{identity['operation']}'. Possible malfunctioning either in the acquisition system or in the machine!"

    config.MAILER.send_mail(object, alert)



def store_datapoint(new_datapoint):

    with open(config.NEW_DATAPOINT_PATH, "w") as json_file:
        json.dump(new_datapoint, json_file, indent=1) 

    with open(config.HISTORICAL_DATA_PATH, "r") as file:
        historical=json.load(file)
    historical=pd.DataFrame(historical)
    historical=pd.concat([historical, pd.DataFrame([new_datapoint])], ignore_index=True)

    with open(config.NEW_DATAPOINT_PATH, "w") as file:
        json.dump(historical.to_dict(), file, indent=1)

