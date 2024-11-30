def get_datapoint():
    # In some manner receives data point as a dictionary of form
    ''' datapoint={timestamp: datetime, 
        isset_id: string,
        name: string,
        kpi: string,
        operation: string,
        sum: float
        avg: float,
        min: float,
        max: float,
        var: float}'''
    #for example

    datapoint = {
    'timestamp': 'timepoint',
    'isset_id': 'ast-yhccl1zjue2t',
    'name': 'metal_cutting',
    'kpi': 'time',
    'operation': 'working',
    'sum': float, 
    'avg': float,
    'min': float,
    'max': float,
    'var': float}
    
    return datapoint


def get_historical_data(machine_name, asset_id, kpi, operation, timestap_start, timestamp_end):
    # In some manner receives data frame filtered from the database in format dataframe
    #Maybe we can define that if we give timestap_start = None, timestamp_end = None,
    #they have to return us x values in the past starting from the last stored point

    return filtered_dataframe 


def send_alert(anomaly_identity):
    
    # In some manner calls the alert function and sends the identity

    return None


def store_datapoint(new_datapoint):
    new_datapoint.to_json('new_datapoint.json', orient='records', lines=True)
    # In some manner gives the new_datapoint dictionary to the database, so they can store it

    return None