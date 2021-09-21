#!/usr/bin/env python3
# Author: Phuthipong, 2021
# Organization: LASS UMASS Amherst
################
# IMPORTANT NOTE: more information about data preprocessing and hyperparameter tuning can be found in the paper.
#################

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from sklearn.ensemble import RandomForestRegressor
# import libraries
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import stats
import matplotlib.pyplot as plt
from calendar import month_name
from calendar import monthrange
from scipy.stats import rankdata

# run
def create_hourly_lstm(X_train, y_train,breg = 0.0, dropout = 0.2, lr = 0.0005, recurrent = 'hard_sigmoid', RS = False, f_bias = 0, bs = 1, decay = 0.001):

    model = Sequential()
    model.add(LSTM(128,input_shape=(X_train.shape[1], X_train.shape[2]),kernel_initializer='glorot_uniform'
                   , bias_regularizer = regularizers.l2(breg), return_sequences=True, recurrent_activation = recurrent
                   , return_state = RS, unit_forget_bias = f_bias, use_bias = bs))
    model.add(LSTM(96, dropout = dropout, return_sequences = True,recurrent_activation = recurrent
                   , return_state = RS, unit_forget_bias = f_bias, use_bias = bs))
    model.add(LSTM(72,recurrent_activation = recurrent, use_bias = bs))
    # output is 24
    model.add(Dense(24))

    # Compiling the LSTM
    adam = Adam(lr = lr,decay = decay)
    model.compile(loss='mae', optimizer='adam')

    return model

if __name__ == '__main__':

    pass




    

