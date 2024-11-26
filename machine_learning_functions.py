from river import drift
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ______________________________________________________________________________________________
# This function takes in input the time serie specific for a feature of a determined machine and
# KPI. It computes the potential drift points present in the given time range and returnes True
# if a drift was detected on the last time stamp.

def ADWIN_drift(dataframe, delta=0.002, clock=10):

    features = ['sum', 'avg','min', 'max', 'var']
    # Initialize variable drift as False
    drift_presence = False

    for feature_name in features:
        # Check if the column exists in the DataFrame
        if feature_name in dataframe.columns:

            feature = dataframe[feature_name]

            # Create ADWIN detector
            adwin = drift.ADWIN(delta=delta, clock=clock)

            # Store drift points
            drift_points = []

            # Iterate through the values in the second column and detect drift
            for i, value in enumerate(feature):
                adwin.update(value)
                if adwin.drift_detected:  # if drift is detected, store index
                    drift_points.append(i)
                    print(f'Change detected at index {i}')
            
            # Check if the last value in the time series is where the drift was detected
            if drift_points and drift_points[-1] == len(feature) - 1:
                drift_presence = True

    return drift_presence

# ______________________________________________________________________________________________
# This class is the one responsible for the training and prediction of anomalies. For the training part 
# it will return the trained model for the specific identity; whereas for the prediction part, it will 
# take a single data point in input and return the prediction.

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
class AnomalyDetector:
    def __init__(self):
        self.features=['sum', 'avg', 'min', 'max', 'var']
        
    def train(self, hist_ts):
        train_set=pd.DataFrame(hist_ts)[self.features]
        s=[]
        cc=np.arange(0.01, 0.5, 0.01)
        for c in cc:
            model = IsolationForest(n_estimators=200, contamination=c)
            an_pred=model.fit_predict(train_set)
            s.append(silhouette_score(train_set, an_pred))
        optimal_c=cc[np.argmax(s)]
        #print(optimal_c)
        model = IsolationForest(n_estimators=200, contamination=optimal_c)
        model.fit_predict(train_set)
        return model 
    
    def predict(self, x, model):
        dp=pd.DataFrame(x[self.features]).T
        anomaly=model.predict(dp)
        if anomaly==-1:
            anomaly='Anomaly'
        else:
            anomaly='Normal'
        return anomaly