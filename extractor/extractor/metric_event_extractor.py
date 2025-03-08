import json
import os
import pandas as pd


def extract_metric_events(pod_host: str, kpi_dic: dict, metric_detector: dict):
    """extract events using 3-sigma from 
        different metric dataframe
    """
    events = []
    for kpi, df in kpi_dic.items():
        df.fillna(0, inplace=True)
        df.sort_values(by=['timestamp'], inplace=True, ascending=True)

        times = df['timestamp'].values
        if len(df)==0:
            continue
        # detect anomaly using 3-sigma
        ab_idx, ab_direction = k_sigma(
            detector=metric_detector[kpi],
            test_arr=df['value'].values,
            k=3,
        )
        
        if ab_idx != -1:
            ab_t = times[ab_idx]
            splits = pod_host.split('_')
            pod, host = splits[0], splits[1]
            events.append([ab_t, pod, host, kpi, ab_direction])
            
    # sort by timestamp
    sorted_events = sorted(events, key=lambda e:e[0])
    # remove timestamp
    sorted_events = [e[1:] for e in sorted_events]
    return sorted_events



def k_sigma(detector, test_arr, k=3):
    mean = detector[0]
    std = detector[1]
    up, lb=mean+k*std, mean-k*std

    for idx, v in enumerate(test_arr.tolist()):
        if v > up:
            return idx, 'up'
        elif v < lb:
            return idx, 'down'
    
    return -1, None
