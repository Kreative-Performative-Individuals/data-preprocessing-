from infoManager import data_path
import json
import pandas as pd

def get_datapoint(i):
    # In some manner receives data point as a dictionary of form

    with open(data_path, "r") as json_file:
        data = json.load(json_file)
        stream_data=data[1]
    i=10
    datapoint = {
    'timestamp': stream_data['time'][i],
    'isset_id': stream_data['time'][i],
    'name': stream_data['time'][i],
    'kpi': stream_data['time'][i],
    'operation': stream_data['time'][i],
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

def send_alert(anomaly_identity):
    
    # In some manner calls the alert function and sends the identity

    return None


def store_datapoint(new_datapoint):
    new_datapoint.to_json('new_datapoint.json', orient='records', lines=True)
    # In some manner gives the new_datapoint dictionary to the database, so they can store it

    return None