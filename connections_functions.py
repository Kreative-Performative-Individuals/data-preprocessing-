import json
import pandas as pd

def get_datapoint(i):
    # In some manner receives data point as a dictionary of form

    with open('synthetic_data.json', "r") as json_file:
        data = json.load(json_file)
        stream_data=data[1]
    i=10
    datapoint = {
    'time': stream_data['time'][i],
    'asset_id': stream_data['asset_id'][i],
    'name': stream_data['name'][i],
    'kpi': stream_data['kpi'][i],
    'operation': stream_data['operation'][i],
    'sum': stream_data['sum'][i], 
    'avg': stream_data['avg'][i],
    'min': stream_data['min'][i],
    'max': stream_data['max'][i],
    'var': stream_data['var'][i]}
        
    return datapoint


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
    #for now the historical data contains only one timeseries with identity: id={'asset_id':  'ast-yhccl1zjue2t',
    #                                                                            'name': 'metal_cutting',
    #                                                                            'kpi': 'time',
    #                                                                            'operation': 'working'}
    # time should be expressed as: '2026-02-04 00:00:00+00:00' and they are available from 2023-05-20 to 2026-02-02
    with open(data_path, "r") as json_file:
        data = json.load(json_file)
    
    historical_data=pd.DataFrame(data[0])

    if timestamp_start == -1 and timestamp_end ==-1:
        timestamp_start = historical_data['time'][ len(historical_data['time']) - 50]
        timestamp_end = historical_data['time'][ len(historical_data['time']) - 1]

    i_start = historical_data[historical_data['time'] == timestamp_start].index[0]
    i_end = historical_data[historical_data['time'] == timestamp_end].index[0]
    historical_data= historical_data[
    (historical_data['name'] == machine_name) & 
    (historical_data['asset_id'] == asset_id) & 
    (historical_data['kpi'] == kpi) & 
    (historical_data['operation'] == operation) & 
    (historical_data.index >= i_start) & 
    (historical_data.index <= i_end)
]
    return historical_data

def send_alert(identity, type, counter=None, probability=None): #the identity returns the type of Kpi and machine for which the anomaly/nan values
                                        # have been detected, type is 'Anomaly' or 'Nan', counter (is the number of consecutive days in
                                        # which we have detected nan) is None if type = 'Anomaly'
    if type == 'Anomaly':
        alert = f"Alert anomaly in {identity['name']} - {identity['asset_id']} - {identity['kpi']} - {identity['operation']}! The probability that this anomaly is correct is {probability}."
    else: 
        alert = f"It has been {counter} days that {identity['name']} - {identity['isset_id']} returns NaN values in {identity['kpi']} - {identity['operation']}. Possible malfunctioning either in the acquisition system or in the machine!"

    # Insert the part to send the alert to GUI for the screen visualization to the user


def store_datapoint(new_datapoint):
    # Write the list of dictionaries to a JSON file
    with open('new_datapoint.json', "w") as json_file:
        json.dump(new_datapoint, json_file, indent=1) 
        # In some manner gives the new_datapoint dictionary to the database, so they can store it
