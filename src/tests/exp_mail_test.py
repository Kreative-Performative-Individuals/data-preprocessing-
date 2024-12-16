import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.app.connection_functions import get_datapoint
import pickle
import json

c = 0
new_datapoint = get_datapoint(c)  # CONNECTION WITH API
print(f"\n{new_datapoint}")

with open('data/store.pkl', 'rb') as file:
    data = pickle.load(file)
    print(json.dumps(data, indent=4))