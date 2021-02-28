# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:17:34 2021

@author: Steve
"""


import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import date
import time
import os
import csv

from sklearn import preprocessing


#Set our working directory
os.chdir(r'C:\Users\Steve\Desktop\crypto_forecast')
os.getcwd()

key = '2282efa0e325e59baf11b4b21a26f2907c29c9fb9d6a7287792330002df79d9c'


TODAY = str(date.today())

def get_crypto_data(coin, limit):
    #get real-time prices from the cryptocompare API
    call = requests.get(f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit={limit}&api_key=' + key).json()
    columns = [['Date', 'Open', 'High', 'Low', 'Close']]
    rows = []
    #Extract data that we want from API. Change timestamp to date.
    for values in call['Data']['Data']:
        stamps = values['time']
        date = time.strftime('%m/%d/%Y', time.localtime(stamps))
        opn = values['open']
        high = values['high']
        low = values['low']
        closing = values['close']
        rows.append((date,opn, high, low, closing))
    #Create dataframe 
    filename = f'./{coin}.csv'
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(columns)
        csvwriter.writerows(rows)
    
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=True)
    #Create extra column with normalized values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df[['Open']])
    df['Value_norm'] = pd.DataFrame(x_scaled)
    
    return df
    
    
ada = get_crypto_data('ADA', 500)  
btc = get_crypto_data('BTC', 500)  
eth = get_crypto_data('ETH', 500)

