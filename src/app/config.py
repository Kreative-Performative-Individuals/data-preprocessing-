import os

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYNTHETIC_DATA_PATH = os.path.join(PROJECT_FOLDER, os.path.join("data", "synthetic_data.json"))
STORE_DATA_PATH = os.path.join(PROJECT_FOLDER, os.path.join("data", "store.json"))
NEW_DATAPOINT_PATH = os.path.join(PROJECT_FOLDER, os.path.join("data", "new_datapoint.json"))
