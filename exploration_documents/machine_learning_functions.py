from river import drift
import pandas as pd
from exploration_documents.infoManager import features, identity, b_length, get_batch

''''
________________________________________________________________________________________________________
FUNCTIONS FOR DRIFT DETECTION
________________________________________________________________________________________________________
'''
# ______________________________________________________________________________________________
# This function takes in input the time serie specific for a feature of a determined machine and
# KPI. It computes the potential drift points present in the given time range and returnes two 
# arguments, the first takes the value False if no drift was detected or True if there is some 
# drift, while the second returns the drift points. 

def ADWIN_drift(x, delta=0.002, clock=10):

    for f in features:

        # Check if the column exists in the DataFrame
        batch = get_batch(x, f)
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

''''
________________________________________________________________________________________________________
FUNCTIONS FOR ANOMALY DETECTION
________________________________________________________________________________________________________
'''
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
        return 
    
''''
________________________________________________________________________________________________________
FUNCTIONS FOR FORECASTING ALGORITHM
________________________________________________________________________________________________________
'''

# Import the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import optuna


def create_sequences(data, tau):
    # Function to create sequences in correct format for the TDNN
    # INPUT:
    # - data: the time series
    # - tau: the length of the sliding window in input to the TDNN
    # OUTPUT:
    # - sequences: sequence in format: (num_sequences, tau)

    num_sequences = len(data) - tau + 1  # Number of sequences
    sequences = np.zeros((num_sequences, tau))  # Initialize matrix
    for i in range(num_sequences):
        sequences[i] = data[i:i+tau]
    return sequences


def split_data(x_data, y_data, train_size=0.8, val_size=0.1, test_size=0.1):
    # Function to split data into training, validation, and test sets
    # INPUT:
    # - x_data: input data set
    # - y_data: target data set
    # train_size = 0.8, val_size = 0.1 and test_size = 0.1: splitting points
    # OUTPUT:
    # - x_train, x_val, x_test, y_train, y_val, y_test: splitted sets
    assert train_size + val_size + test_size == 1, "The splits should sum to 1"

    # Split into training, validation and test sets
    train_val_size = int(len(x_data) * train_size)  # 80% for training + validation
    x_train_val, x_test = x_data[:train_val_size], x_data[train_val_size:]
    y_train_val, y_test = y_data[:train_val_size], y_data[train_val_size:]

    # Further split the train_val set into training and validation sets
    train_size = int(len(x_train_val) * (train_size / (train_size + val_size)))  # 80% of 80% for training
    x_train, x_val = x_train_val[:train_size], x_train_val[train_size:]
    y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_TDNN(hidden_units, lr):
    # Function that creates the TDNN model
    # The model comprises:
    # - 3 hidden Dense layers of hidden_units neuron with relu as activation function
    # - an output layer
    # The loss is calculated with MSE (Mean Square Error) and the optimizer is Adam
    # INPUT:
    # - hidden_units: number of neurons in the layer
    # - lr: learning rate
    # OUTPUT:
    # - model: the model
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr))
    return model


def training_TDNN(TDNN_model, x_train, y_train, x_val, y_val, epochs):
    # Function to train the TDNN model
    # INPUT
    # - TDNN_model: the TDNN model
    # - x_train, y_train, x_val, y_val: input and target training and validation sets
    # - epochs: number of epochs
    # OUTPUT:
    # - loss_validation: the loss of the trained model that we want to minimize
    history = TDNN_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), verbose=0)
    print(history.history.keys())
    loss_training = history.history['loss']
    print('Loss training: ', loss_training[-1])
    loss_validation = history.history['val_loss']
    print('Loss validation: ', loss_validation[-1])
    return loss_validation[-1]


def objective_TDNN(trial, time_series):
    # Function that uses Optuna for hyperparameters optimization
    # HYPERPARAMETERS:
    # - tau: length of input sliding window to TDNN
    # - epochs: number of epochs to train TDNN
    # - lr: learning rate
    # - hidden_units: number of neurons in TDNN layers
    # INPUT:
    # - trial: number of trials of the study to search for hyperparameters
    # - time series: time series we want to train
    # OUTPUT:
    # - val_loss: loss of the validation that we want to minimize

    # Set hyperparameters ranges
    tau = trial.suggest_categorical('tau', [7 ,14 , 21])
    epochs = trial.suggest_int('epochs', 50, 150, step=10)
    lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
    hidden_units = trial.suggest_int('hidden_units', 50, 250)
    TDNN_model = create_TDNN(hidden_units, lr)

    # Create sequences for the model
    sequences = create_sequences(time_series, tau)
    x_data = sequences[:, :-1]  # All but the last value as features
    y_data = time_series[tau-1:]  # The corresponding targets

    #print(x_data.shape)
    #print(y_data.shape)

    # Split data into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_data, y_data)

    # Compute mean and std from training data
    x_mean = np.mean(x_train)
    x_std = np.std(x_train)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    # Normalize training and test data with mean and variance of the training
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    # Reshape the input data to (1, num_sequences, tau)
    x_train = np.expand_dims(x_train, axis=0)  # Shape (1, num_sequences, tau)
    x_val = np.expand_dims(x_val, axis=0)  # Shape (1, num_sequences, tau)

    # Reshape target data to (1, num_sequences)
    y_train = np.expand_dims(y_train, axis=0)  # Shape (1, num_sequences)
    y_val = np.expand_dims(y_val, axis=0)  # Shape (1, num_sequences)
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_val.shape)
    #print(y_val.shape)

    # Train the model and return the validation loss
    val_loss = training_TDNN(TDNN_model, x_train, y_train, x_val, y_val, epochs)
    return val_loss


def tdnn_forecasting_training(time_series, n_trials=10):
    # Function that creates a study for hyperparameter optimization
    # and validates the TDNN using the best hyperparameters found
    # INPUT:
    # - time series: time series we want to train and find best model and hyperparameters
    # - n_trials: number of trials for hyperparameter search
    # OUTPUT:
    # - best_model_TDNN: model of TDNN with best hyperparameters
    # - best_params: dictionary comprising best hyperparameters ['tau', 'lr', 'epochs', 'hidden_units']
    # - stats: array comprising [x_mean, x_std, y_mean, y_std] which are needed for proper normalization

    # Create study and save best params
    TDNN_study = optuna.create_study(direction='minimize')
    TDNN_study.optimize(lambda trial: objective_TDNN(trial, time_series), n_trials=n_trials)
    best_params = TDNN_study.best_params
    #print('Best Hyperparameters:', best_params)
    tau = best_params['tau']
    epochs = best_params['epochs']
    hidden_units = best_params['hidden_units']
    lr = best_params['lr']

    # Create model with best hyperparameters
    best_model_TDNN = create_TDNN(hidden_units, lr)

    # Split time_series into input and target
    sequences = create_sequences(time_series, tau)
    x_data = sequences[:, :-1]  # All but the last value as features
    y_data = time_series[tau-1:]  # The corresponding targets

    # Split data into training, validation, and test sets
    x_training, x_val, x_test, y_training, y_val, y_test = split_data(x_data, y_data)

    # Compute mean and std from training data
    x_mean = np.mean(x_training)
    x_std = np.std(x_training)
    y_mean = np.mean(y_training)
    y_std = np.std(y_training)
    stats = np.array([x_mean, x_std, y_mean, y_std])

    # Normalize training and test data with training stats
    x_training = (x_training - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_training = (y_training - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Reshape input data to (1, num_sequences, tau)
    x_training = np.expand_dims(x_training, axis=0)  # Shape (1, num_sequences, tau)
    x_test = np.expand_dims(x_test, axis=0)  # Shape (1, num_sequences, tau)

    # Reshape target data to (1, num_sequences)
    y_training = np.expand_dims(y_training, axis=0)  # Shape (1, num_sequences)
    y_test = np.expand_dims(y_test, axis=0)  # Shape (1, num_sequences)

    # Train the model
    history = best_model_TDNN.fit(x_training, y_training, epochs=epochs, verbose=0)
    # print(history.history['loss'][-1])

    # Predict on training and test data
    y_pred_training = best_model_TDNN.predict(x_training).reshape(-1)
    y_pred_test = best_model_TDNN.predict(x_test).reshape(-1)

    # calculate MSE
    TDNN_test_MSE = best_model_TDNN.evaluate(x_test, y_test)
    # print('Test MSE: ', TDNN_test_MSE)

    # Denormalize predictions and targets for plotting
    #y_pred_training = y_pred_training * y_std + y_mean
    #y_pred_test = y_pred_test * y_std + y_mean
    #y_training = y_training * y_std + y_mean
    #y_test = y_test * y_std + y_mean

    # Plot the results
    #plt.figure(figsize=(12, 10))
    #plt.subplot(2, 1, 1)
    #plt.plot(y_training.reshape(-1), label='Target')
    #plt.plot(y_pred_training, label='Predicted')
    #plt.title('Predicted Training vs. Target Training')
    #plt.legend()
    #plt.xlabel('Time')
    #plt.subplot(2, 1, 2)
    #plt.plot(y_test.reshape(-1), label='Target')
    #plt.plot(y_pred_test, label='Predicted')
    #plt.title('Predicted Test vs. Target Test')
    #plt.xlabel('Time')
    #plt.legend()
    #plt.show()

    return [best_model_TDNN, best_params, stats]



def tdnn_forecasting_prediction(model, tau, time_series, num_predictions, stats):
    # Function that uses the trained model to predict num_predictions in the future
    # INPUT:
    # - model: TDNN best model after training
    # - tau: length of input sliding window which can be retrieved from best_params['tau']
    # - num_predictions: number of future step to predict
    # - stats: list with statistics to normalize the time series

    x_mean, x_std, y_mean, y_std = stats
    sequences = create_sequences(time_series, tau)
    initial_window = sequences[-1, :-1]  # Use the last sequence as the initial window
    predictions = []
    current_window = (initial_window - x_mean) / x_std  # Normalize the input window

    for _ in range(num_predictions):
        # Predict the next value
        # Reshape the input to match the model's expected input shape (1, 1, tau-1)
        next_value_norm = model.predict(current_window.reshape(1, 1, -1))[0, 0]
        next_value = next_value_norm * y_std + y_mean  # Denormalize the prediction
        predictions.append(next_value)

        # Update the current window for the next prediction
        current_window = np.append(current_window[1:], (next_value - x_mean) / x_std)

    return predictions