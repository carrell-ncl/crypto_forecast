# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:28:10 2021

@author: Steve
"""
import os

#Set our working directory
os.chdir(r'C:\Users\Steve\Desktop\crypto_forecast')
os.getcwd()

from crypto_getprices import get_crypto_data
from processing import get_train_test_val_df
from train_model import train_model
from get_forcasts import get_forcast

btc = get_crypto_data('BTC', 500)  

train, test, val = get_train_test_val_df(btc)

my_model , history = train_model(train, test, val, 'BTC')

get_forcast(train, test, my_model, 30, 'BTC')
