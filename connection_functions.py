import json
import pandas as pd
import os 

def get_datapoint(i):
    

    with open(os.path.join(os.getcwd(), os.path.join("initialization", "original_adapted_dataset.json")), "r") as file:
        stream=json.load(file)

    stream=pd.DataFrame(stream)[38400:]
    datapoint = stream.iloc[i].to_dict()
        
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
from datetime import datetime, timedelta

def get_historical_data(machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end):
    #for now the historical data contains only one timeseries with identity: id={'asset_id':  'ast-yhccl1zjue2t',
    #                                                                            'name': 'metal_cutting',
    #                                                                            'kpi': 'time',
    #                                                                            'operation': 'working'}
    # time should be expressed as: '2026-02-04 00:00:00+00:00'
    # with open("synthetic_data.json", "r") as json_file:
    #     data = json.load(json_file)
    
    # historical_data=pd.concat([pd.DataFrame(data[0]), pd.DataFrame(data[1])], axis=0)
    with open(os.path.join(os.getcwd(), os.path.join("initialization", "historical_dataset.json")), "r") as file:
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

    # if timestamp_start == -1 and timestamp_end ==-1:
    #     i_end = historical_data['time'][len(historical_data['time'])-1]
    #     i_start=i_end-100
    # elif timestamp_start == -1:
    #     i_end = historical_data[historical_data['time'] == timestamp_end].index[0]
    #     i_start=i_end-100
    #     # print(i_start)
    #     # print(i_end)
    if timestamp_end ==-1:
        timestamp_end=historical_data['time'].iloc[-1]

    if timestamp_start==-1:
        timestamp_start=timestamp_end - timedelta(days=100)       

    historical_data= historical_data[
    (historical_data['time'] >= timestamp_start) & 
    (historical_data['time'] <= timestamp_end)]
    
    historical_data['time']=historical_data['time'].astype(str)

    return historical_data


from notification.mail_sender import MailSender
def send_alert(identity, type, counter=None, probability=None): #the identity returns the type of Kpi and machine for which the anomaly/nan values
                                        # have been detected, type is 'Anomaly' or 'Nan', counter (is the number of consecutive days in
                                        # which we have detected nan) is None if type = 'Anomaly'
    if type == 'Anomaly':
        object='KPI - Anomaly alert'
        alert = f"Alert anomaly in machine: '{identity['name']}' - asset: '{identity['asset_id']}' - kpi: '{identity['kpi']}' - operation: '{identity['operation']}'! The probability that this anomaly is correct is {probability}%."
    else:
        object='Malfunctioning alert' 
        alert = f"It has been {counter} days that machine: '{identity['name']}' - asset: '{identity['asset_id']}' returns NaN values in kpi: '{identity['kpi']}' - operation: '{identity['operation']}'. Possible malfunctioning either in the acquisition system or in the machine!"
    #sender = MailSender('mcaponio28@libero.it', 'p@sswordLiber0', 'mcaponio28@gmail.com')

    #sender.send_mail(object, alert)
    print(alert)
    # Insert the part to send the alert to GUI for the screen visualization to the user


def store_datapoint(new_datapoint):
    #print(f'{new_datapoint} \n')
    # Write the dictionary (the new datapoint)
    with open('new_datapoint.json', "w") as json_file:
        json.dump(new_datapoint, json_file, indent=1) 
        # In some manner gives the new_datapoint dictionary to the database, so they can store it
    # with open('synthetic_data.json', "r") as json_file:
    #     data=json.load(json_file)
    #     historical_data=data[0]
    #     stream_data=data[1]
    
    # stream_data['time'][i]=new_datapoint['time']
    # stream_data['name'][i]=new_datapoint['name']
    # stream_data['asset_id'][i]=new_datapoint['asset_id']
    # stream_data['kpi'][i]=new_datapoint['kpi']
    # stream_data['operation'][i]=new_datapoint['operation']
    # stream_data['sum'][i]=new_datapoint['sum']
    # stream_data['max'][i]=new_datapoint['max']
    # stream_data['min'][i]=new_datapoint['min']
    # stream_data['avg'][i]=new_datapoint['avg']
    # stream_data['var'][i]=new_datapoint['var']
    # stream_data['status'][i]=new_datapoint['status']

    with open(os.path.join(os.getcwd(), os.path.join("initialization", "historical_dataset.json")), "r") as file:
        historical=json.load(file)
    historical=pd.DataFrame(historical)
    historical=pd.concat([historical, pd.DataFrame([new_datapoint])], ignore_index=True)

    with open(os.path.join(os.getcwd(), os.path.join("initialization", "historical_dataset.json")), "w") as file:
        json.dump(historical.to_dict(), file, indent=1)

    # with open('synthetic_data.json', "w") as json_file:
    #     json.dump([historical_data, stream_data], json_file, indent=1)

