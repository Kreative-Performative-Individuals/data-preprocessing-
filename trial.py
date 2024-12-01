from infoManager import data_path
import json
from connections_functions import get_historical_data

dataframe=get_historical_data('metal_cutting','ast-yhccl1zjue2t', 'time','working', '2023-05-20 00:00:00+00:00', '2023-06-03 00:00:00+00:00')
print(dataframe)