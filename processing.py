# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:57:39 2021

@author: Steve
"""


import pandas as pd


def get_train_test_val_df(main_df):
    
    #Create our split dfs ready for training
    train_len = len(main_df)-30
    
    train_set = main_df['Open'][:train_len]
    train_set = pd.DataFrame(train_set)
    
    test_set = main_df['Open'][train_len:]
    test_set = pd.DataFrame(test_set)
    
    val_set = main_df['Open'][train_len-60:]
    val_set = pd.DataFrame(val_set)

    return train_set, test_set, val_set


