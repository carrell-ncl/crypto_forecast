# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:24:52 2021

@author: Steve
"""

from datetime import datetime
from sklearn import preprocessing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback
import warnings

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            
callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.008, verbose=1)
]

def get_forcast (train_df, test_df, val_df, coin):
    min_max_scaler = preprocessing.MinMaxScaler()
    training_set_scaled = min_max_scaler.fit_transform(train_df)
    testing_set_scaled = min_max_scaler.fit_transform(val_df)

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    
    for i in range(60, len(training_set_scaled)):
        x_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
        
    for i in range(60, len(testing_set_scaled)):
        x_val.append(testing_set_scaled[i-60:i, 0])
        y_val.append(testing_set_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    
    
    #Reshape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    
    #Initialisze
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    #Set early stoppage when validation loss stops decreasing for 5 epochs
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) 
    
    history = model.fit(x_train, y_train, epochs = 300, batch_size=32, validation_data=(x_val, y_val), callbacks=callbacks)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    model.save_weights(f'{coin}weights{dt_string}.h5')
    
    return model, history