#!/usr/bin/env python3
# Author: Phuthipong, 2021
# Organization: LASS UMASS Amherst

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
import math
import matplotlib.pyplot as plt 

def create_daily_lstm(breg = 0.1, dropout = 0.2 , lr = 0.001, recurrent = 'hard_sigmoid', RS = False, f_bias = 0.1, bs = 0, stateful=True):
    model = Sequential()
    model.add(LSTM(32,input_shape=(X_train.shape[1], X_train.shape[2]),kernel_initializer='glorot_uniform', bias_regularizer = regularizers.l2(breg),return_sequences=True,recurrent_activation = recurrent , return_state = RS,unit_forget_bias = f_bias, use_bias = bs, stateful=stateful))
    model.add(LSTM(16, dropout = dropout, return_sequences = False,recurrent_activation = recurrent ,return_state = RS,unit_forget_bias = f_bias, use_bias = bs))
    model.add(Dense(1))

    # Compiling the LSTM
    adam = Adam(lr = lr,decay = 0.0001)
    model.compile(loss='mae', optimizer='adam')
    return model

if __name__ == '__main__':
    pass


