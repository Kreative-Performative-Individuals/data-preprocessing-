from river import drift
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ______________________________________________________________________________________________
# This function takes in input the time serie specific for a feature of a determined machine and
# KPI. It computes the potential drift points present in the given time range and returnes two 
# arguments, the first takes the value False if no drift was detected or True if there is some 
# drift, while the second returns the drift points. 

def ADWIN_drift(time_serie, delta=0.002, clock=10):

    # Create ADWIN detector
    adwin = drift.ADWIN(delta=delta, clock=clock)

    # Initialize variable drift as False
    drift = False

    # Store drift points
    drift_points = []

    # Iterate through the values in the second column and detect drift
    for i, value in enumerate(time_serie.iloc[:, 1]):
        adwin.update(value)
        if adwin.drift_detected:                       # if drift is detected, store index
            drift_points.append(i)
            print(f'Change detected at index {i}')
            drift = True
            

    return drift, drift_points

## it should only notificate for new detected drift! 