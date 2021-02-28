# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:01:44 2021

@author: Steve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_forcast (train_df, test_df, model, no_days, coin):
    real_prices = test_df.values
    
    dataset_total = pd.concat((train_df, test_df), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test_df) - 60:].values
    
    inputs = inputs.reshape(-1,1)
    sc = preprocessing.MinMaxScaler()
    inputs = sc.fit_transform(inputs)
    
    predictions = []
    predictions_transformed = []
    temp = list(inputs[0:60,0])
    
    for i in range(60, 60+no_days):
        
        temp_array = np.array(temp)
        
        temp_array =  np.reshape(temp_array, (1, temp_array.shape[0], 1))
        pred = model.predict(temp_array)
        temp = temp[1:]
        temp.append(pred[0][0])
        predictions.append(pred[0][0])
        pred2 = sc.inverse_transform(pred)
        predictions_transformed.append(pred2[0])
    
    
    
    plt.figure(figsize=(16,6))
    plt.plot(real_prices, color = 'red', label = f'Real {coin} Price')
    plt.plot(predictions_transformed, color = 'blue', label = f'Predicted {coin} Price')
    plt.title(f'{coin} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{coin} Price')
    plt.legend()
    plt.show()
    



