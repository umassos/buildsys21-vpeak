#!/usr/bin/env python3
# Author: Phuthipong, 2021
# Organization: LASS UMASS Amherst

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import Counter
from calendar import monthrange
from scipy import stats
from scipy.stats import rankdata
from statistics import mean 
from sklearn.metrics import confusion_matrix

def run_vpeak_prediction(k, delta, threshold_list, cdf, y_pred_month, y_true_rank, alpha=0.0295, beta=0.01, outlier_check=True, debug=False):
    pred_peak = []
    gt_peak = []
    acc_list = []
    for month in range(1, 13):
        peak_day_picked = []
        peak_picked = []
        historical_peak = 0
        picked = 0
        threshold = threshold_list[month-1]

        plt_thr = []
        plt_dmd = []

        for idx, daily_load in enumerate(y_pred_month[month]):
            # adjust threshold every 3 days
            if idx %5 == 0 and idx > 0:
                # should we increase or decrease the threshold
                # if already pick more than density, increase threshold
                current_weight = picked/(k+delta)
                if current_weight > cdf[month][idx-1]:
                    threshold += threshold*alpha
                elif current_weight < cdf[month][idx-1]:
                    # if less, decrease
                    threshold -= threshold*beta

            if picked < (k+delta):
                if daily_load >= threshold:
                    peak_day_picked.append(True)
                    peak_picked.append(daily_load)
                    picked += 1
                else:
                    is_outlier = False
                    if idx > 9:
                        past_pred = y_pred_month[month][0:idx-1]
                        # outlier in past
                        if daily_load > (np.mean(past_pred) + np.std(past_pred)*1.5):
                            is_outlier = True

                    if is_outlier and outlier_check:
                        peak_day_picked.append(True)
                        peak_picked.append(daily_load)
                        picked += 1
                    else:
                        peak_day_picked.append(False)

            else:
                peak_day_picked.append(False)

            if daily_load > historical_peak:
                historical_peak = daily_load

            plt_thr.append(threshold)
            plt_dmd.append(daily_load)

        pred_peak += peak_day_picked
        gt_peak += [(r <= k) for r in y_true_rank[month]]
        acc = get_accuracy(y_true_rank[month], peak_day_picked, k)
        if debug:
            plt.plot(plt_dmd, label="pred load")
            plt.plot(plt_thr, label="threshold")
            plt.plot(y_true_month[month], label="gt load")
            plt.legend()
            plt.show()
            plt.plot(peak_day_picked)
            plt.show()
            print(month)
            print(sum(peak_day_picked))
            print(acc)
        acc_list.append(acc)

    tn, fp, fn, tp = confusion_matrix(gt_peak, pred_peak).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    acc = (tp+tn)/(tn + fp + fn + tp)
    print("k: {}, delta: {}, p: {}, r: {}, acc: {}".format(k, delta, precision, recall, acc))
#     print("k: {}, delta: {}, acc: {}".format(k, delta, sum(acc_list)/len(acc_list)))
    recall = sum(acc_list)/len(acc_list)
    
    return recall

def find_threshold(historical_df, quantile, y_column):
    # input: at least, one year of data
    # find monthly threshold
    # return: monthly threshold
    threshold_list = []
    for i in range(1, 13):
        month_mask = (historical_df.index.month == i)
        threshold = historical_df[month_mask][y_column].round().quantile(quantile)
        threshold_list.append(threshold)
    
    return threshold_list

def find_cdf(historical_df, y_column, top_k):
    # input: one year of data
    # find peak day that should have picked for each day of the month
    # return: cdf each month
    cdf = {}
    for month in range(1, 13):
        # calculate cdf 
        month_mask = (historical_df.index.month == i)
        d_demand = historical_df[month_mask][y_column].values
        num_of_day = len(d_demand)
        hist, bins = np.histogram(d_demand.argsort()[::-1][:top_k],bins=np.linspace(0,num_of_day,num_of_day+1))
        cdf[month] = hist.cumsum()/top_k

    return cdf

if __name__ == '__main__':
    top_k = 5 # number of top k peak day 
    delta = 3 # additional day for picking
    quantile = 1 - (top_k/30) # to find threshold, can be changed
    thres_list = find_threshold(historical_df, 0.93, "Power")
    cdf = find_cdf(historical_df, y_column, top_k)

    y_pred_month # use LSTM model to predict demand and assign here, expected dimension (12, X) where X is the number of days in each month
    y_true_rank # groundtruth for checking the accuracy
    # alpha and beta can be tuned on historical data
    alpha = 0.03
    beta = 0.01
    acc = run_vpeak_prediction(top_k, delta, thres_list, cdf, y_pred_month, y_true_rank, alpha, beta)
    

