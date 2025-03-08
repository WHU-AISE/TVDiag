import json
import os
import numpy as np

import pandas as pd
import utils.io_util as io_util
import utils.detect_util as d_util


def slide_window(df, win_size):
    sts, ds, err_500_ps, err_400_ps=[], [], [], []
    # df_copy=df.copy()
    # df_copy['slide_duration']=df['duration'].rolling(window=10, min_periods=1).mean()
    df['duration'] = df['end_time']-df['start_time']
    i, time_max=df['start_time'].min(), df['start_time'].max()
    while i < time_max:
        temp_df = df[(df['start_time']>=i)&(df['start_time']<=i+win_size)]
        if temp_df.empty:
            i+=win_size
            continue
        sts.append(i)
        # error_code_n = len(temp_df[~temp_df['status_code'].isin([200, 300])])
        err_500_ps.append(len(temp_df[temp_df['status_code']==500]))
        err_400_ps.append(len(temp_df[temp_df['status_code']==400]))
        ds.append(temp_df['duration'].mean())
        i+=win_size
    return np.array(sts), np.array(ds), np.array(err_500_ps), np.array(err_400_ps)



def extract_trace_events(df: pd.DataFrame, trace_detector: dict):
    """extract events using iforest from 
        trace dataframe
    """
    events = []
    df.sort_values(by=['timestamp'], inplace=True, ascending=True)
    df['operation'] = df['url'].str.split('?').str[0]
    gp = df.groupby(['parent_name', 'service_name', 'operation'])
    events = []

    win_size = 30 * 1000
    # detect events for every call
    for (src, dst, op), call_df in gp:
        name = src + '-' + dst +'-' + op
        test_df = call_df
        test_win_sts, test_durations, err_500_ps, err_400_ps = slide_window(test_df, win_size)
        if len(test_durations) > 0:
            pd_idx = iforest(trace_detector[name]['dur_detector'], test_durations)
            err_500_idx = iforest(trace_detector[name]['500_detector'], err_500_ps)
            err_400_idx = iforest(trace_detector[name]['400_detector'], err_400_ps)

            if pd_idx != -1:
                events.append([test_win_sts[pd_idx], src, dst, op, 'PD'])
            if err_500_idx != -1:
                events.append([test_win_sts[err_500_idx], src, dst, op, '500'])
            if err_400_idx != -1:
                events.append([test_win_sts[err_400_idx], src, dst, op, '400'])
            
    events = sorted(events, key=lambda x: x[0])
    events = [x[1:] for x in events]
    return events


def iforest(detector, test_arr):
    labels = detector.predict(test_arr.reshape(-1,1)).tolist()
    try:
        idx = labels.index(-1)
    except:
        return -1
    return idx