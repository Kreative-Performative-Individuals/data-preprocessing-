from river import drift
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from information import features, identity, infoManager, b_length
# ______________________________________________________________________________________________
# This function takes in input the time serie specific for a feature of a determined machine and
# KPI. It computes the potential drift points present in the given time range and returnes two 
# arguments, the first takes the value False if no drift was detected or True if there is some 
# drift, while the second returns the drift points. 

def ADWIN_drift(x, im, delta=0.002, clock=10):

    for f in features:

        # Check if the column exists in the DataFrame
        batch = im.get_batch(x, f)
        batch1 = batch[:-1]
        batch2 = batch[1:]
        adwin = drift.ADWIN(delta=0.05, clock=10)
        flag1=False
        flag2=False

        for i, value in enumerate(batch1):
            adwin.update(value)
            if adwin.drift_detected and i>b_length-3:
                flag1=True
        for i, value in enumerate(batch2):
            adwin.update(value)
            if adwin.drift_detected and i>b_length-3:
                flag2=True
        if flag1==False and flag2==True:
            return True
        else:
            return False

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