import sys
import pandas as pd
import numpy as np

sys.path.append("C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-")
from dataprocessing_functions import (
    fields,
    features,
    identity,
    check_f_consistency,
    kpi,
    get_batch,
    update_counter,
    imputer,
    get_counter,
    faulty_aq_tol,
    update_batch,
)
from datetime import datetime
from collections import OrderedDict


def validate(x):

    for f in fields:
        x.setdefault(
            f, np.nan
        )  # if some fields is missing from the expected ones, put a nan
    x = dict(
        OrderedDict((key, x[key]) for key in fields)
    )  # order the fields of the datapoint

    # Ensure the reliability of the field time
    if pd.isna(x["time"]):
        x["time"] = datetime.now()

    # Check that there is no missing information in the identity of the datapoint, otherwise we store in the database, labelled 'Corrupted'.
    if any(pd.isna(x.get(key)) for key in identity):
        update_counter(x)
        x["status"] = "Corrupted"
        return x
    # Check if all the features that the datapoint has are nan or missing.
    elif all(pd.isna(x.get(key)) for key in features):
        update_counter(x)
        x["status"] = "Corrupted"
        return x

    # if the datapoint comes here it means that it didn't miss any information about the identity and at least one feature that is not nan.

    x = check_range(
        x
    )  # the flag is to take trace if the datapoint has naturally nans or nans are the result of validation checks.

    # if the datapoint comes here it means that at least one feature value is respecting the range constraint for the specific kpi.
    if x:
        # Check if the features (min, max, sum, avg) satisfy the basic logic rule min<=avg<=max<=sum
        cc = check_f_consistency(x)
        if all(not c for c in cc):  # meaning that no feature respect the logic rule
            update_counter(x)
            x["status"] = "Corrupted"
            return x
        elif all(c for c in cc):  # the datapoint verifies the logic rule.
            # if now there is a nan it could be either the result of the range check or that the datapoint intrinsically has these nans.
            any_nan = False
            for f in features:
                if np.isnan(x[f]):
                    any_nan = True
                    if all(np.isnan(get_batch(x, f))):
                        pass
                    else:
                        update_counter(x)
                        break
            if any_nan == False:
                # it means that the datapoint is consistent and it doesn't have nan values --> it is perfect.
                update_counter(x, True)  # reset the counter.
        else:  # it means that some feature are consistent and some not. Put at nan the not consistent ones.
            for f, c in zip(features, cc):
                if c == False:
                    x[f] = np.nan
            update_counter(x)
        x["status"] = "A/N"
        return x


def check_range(x):

    # Retrieve the specific range for the kpi that we are dealing with
    l_thr = kpi[x["kpi"]][0][0]
    h_thr = kpi[x["kpi"]][0][1]

    for k in features:
        if x[k] < l_thr:
            x[k] = np.nan
        if k in ["avg", "max", "min", "var"] and x[k] > h_thr:
            x[k] = np.nan

    # if after checking the range all features are nan --> corrupted
    if all(np.isnan(value) for value in [x.get(key) for key in features]):
        update_counter(x)
        x["status"] = "Corrupted"
    return x


def check_range_ai(x):
    flag = True  # takes trace of: has the datapoint passed the range check without being changed?
    l_thr = kpi[x["kpi"]][0][0]
    h_thr = kpi[x["kpi"]][0][1]

    for k in features:
        if x[k] < l_thr:
            flag = False
        if k in ["avg", "max", "min", "var"] and x[k] > h_thr:
            flag = False
    return flag


from statsmodels.tsa.holtwinters import ExponentialSmoothing


def predict_missing(batch):
    seasonality = 7
    cleaned_batch = [x for x in batch if not np.isnan(x)]
    if not (all(pd.isna(x) for x in batch)) and batch:
        if len(cleaned_batch) > 2 * seasonality:
            model = ExponentialSmoothing(
                cleaned_batch, seasonal="add", trend="add", seasonal_periods=seasonality
            )
            model_fit = model.fit()
            prediction = model_fit.forecast(steps=1)[0]
        else:
            prediction = np.nanmean(batch)
        return prediction
    else:
        return (
            np.nan
        )  # Leave the feature as nan since we don't have any information in the batch to make the imputation. If the datapoint has a nan because the feature is not definable for it, it will be leaved as it is from the imputator.


# ______________________________________________________________________________________________
# This function is the one managing the imputation for all the features of the data point  receives as an input the new data point, extracts the information


def imputer(x):
    if x:
        if isinstance(x, tuple):
            x = x[0]
            # Because the validated datapoint may exit in the check range with 2 returned values.

        # Try imputation with mean or the HWES model.
        for f in features:
            batch = get_batch(x, f)
            if pd.isna(x[f]):
                x[f] = predict_missing(batch)

        # Check again the consistency of features and the range.
        if check_f_consistency(x) and check_range_ai(x):
            pass
        else:  # It means that the imputed data point has not passed the check on the features and on their expected range.
            # In this case we use the LVCF as a method of imputation since it ensures the respect of these conditiono (the last point in the batch has been preiovusly checked)
            for f in features:
                batch = get_batch(x, f)
                x[f] = batch[-1]

        # In the end update batches with the new data point
        for f in features:
            update_batch(x, f, x[f])

        return x


import json

with open(
    "C:\\Users\\mcapo\\Desktop\\Smart app project\\definitive\\definitivo_3\\transformation_interrupted.json",
    "r",
) as json_file:
    in_data = json.load(json_file)

cleaned_df = pd.DataFrame(in_data[1])
start_i = in_data[2]
df = pd.DataFrame(in_data[0])

import warnings

warnings.filterwarnings("ignore")
length = df.shape[0] // 4
for i in range(start_i, start_i + 4):
    datapoint = df.iloc[i].to_dict()
    old_counter = get_counter(datapoint)
    # print(f'original datapoint: {datapoint}')
    datapoint = validate(datapoint)
    new_counter = get_counter(datapoint)
    if new_counter == old_counter + 1 and new_counter >= faulty_aq_tol:
        id = {key: datapoint[key] for key in identity if key in datapoint}
        f"It has been {new_counter} days (from {datapoint['time']} that {id['name']} - {id['asset_id']} returns NaN values in {id['kpi']} - {id['operation']}. Possible malfunctioning either in the acquisition system or in the machine!"
    if datapoint["status"] != "Corrupted":
        cleaned_datapoint = imputer(datapoint)
    cleaned_df.iloc[i] = cleaned_datapoint

data = [df.to_dict(), cleaned_df.to_dict(), i]
with open(
    "C:\\Users\\mcapo\\Desktop\\Smart app project\\definitive\\definitivo_3\\transformation_interrupted.json",
    "w",
) as json_file:
    json.dump(data, json_file, indent=1)

with open(
    "C:\\Users\\mcapo\\data-preprocessing-\\data-preprocessing-\\store.json", "r"
) as json_file:
    info = json.load(json_file)

with open(
    "C:\\Users\\mcapo\\Desktop\\Smart app project\\definitive\\definitivo_3\\store.json",
    "w",
) as json_file:
    json.dump(info, json_file, indent=1)
