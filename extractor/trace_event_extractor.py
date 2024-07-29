import json
import os
import numpy as np

import pandas as pd
import utils.io_util as io_util
import utils.detect_util as d_util
from utils.time_util import coast_time, time2stamp


def slide_window(df, win_size):
    sts, ds, cd_500_ns, cd_400_ns=[], [], [], []
    # df_copy=df.copy()
    # df_copy['slide_duration']=df['duration'].rolling(window=10, min_periods=1).mean()
    i, time_max=df['timestamp'].min(), df['timestamp'].max()
    while i < time_max:
        temp_df = df[(df['timestamp']>=i)&(df['timestamp']<=i+win_size)]
        if temp_df.empty:
            i+=win_size
            continue
        sts.append(i)
        cd_500_ns.append(len(temp_df[temp_df['status_code']==500]))
        cd_400_ns.append(len(temp_df[temp_df['status_code']==400]))
        ds.append(temp_df['duration'].mean())
        i+=win_size
    return np.array(sts), np.array(ds), np.array(cd_500_ns), np.array(cd_400_ns)


def process_normal(normal_data, win_size, edges):
    normal_data.sort_values(by=['timestamp'], inplace=True, ascending=True)
    normal_data.fillna(0, inplace=True)
    # sources, targets = edges[0], edges[1]
    res = {}
    for caller, callee in edges:
        # t_idx = targets[i]
        # caller, callee = nodes[s_idx], nodes[t_idx]
        con2 = (normal_data['service_name'] == callee) & (normal_data['parent_name'] == caller)
        train_df = normal_data[con2]
        train_sts, train_ds, train_500ns, train_400ns=slide_window(train_df, win_size)

        res[caller+'_'+callee]=(train_sts, train_ds, train_500ns, train_400ns)
    return res


@coast_time
def extract_events(df: pd.DataFrame, normal_res, win_size,
                    edges: tuple):
    """extract events using DBSCAN from 
        trace dataframe

    Args:
        df (pd.DataFrame): trace dataframe
        nodes (list): microservice instances
        edges (tuple): ([], [])

    Returns:
        events: extracted abnormal events
    """
    # sources, targets = edges[0], edges[1]
    events = []
    df.sort_values(by=['timestamp'], inplace=True, ascending=True)
    df.fillna(0, inplace=True)

    edgeset=set()

    for caller, callee in edges:
        # t_idx = targets[i]
        # caller, callee = nodes[s_idx], nodes[t_idx]

        if (caller + callee) in edgeset:
            continue
        else:
            edgeset.add(caller + callee)

        con1 = (df['service_name'] == callee) & (df['parent_name'] == caller)
        
        test_df = df[con1]
        
        if len(test_df) != 0:
            train_sts, train_ds, train_500ns, train_400ns=normal_res[caller+'_'+callee]
            test_sts, test_ds, test_500ns, test_400ns=slide_window(test_df, win_size)

            if len(test_ds) != 0:
                ab_points, labels = d_util.IsolationForest_detect(
                    train_arr=train_ds,
                    test_arr=test_ds
                )
                ab_points = ab_points.tolist()
                # sts = test_df['start_time'].values
                ab_t = test_sts[labels==-1]
                if len(ab_points) > 0:
                    events.append([ab_t[0], callee, caller, 'PD'])

            if len(test_500ns) != 0:
                ab_points, labels = d_util.IsolationForest_detect(
                    train_arr=train_500ns,
                    test_arr=test_500ns
                )
                ab_points = ab_points.tolist()
                ab_t = test_sts[labels==-1]
                if len(ab_points) > 0:
                    events.append([ab_t[0], callee, caller, '500'])

            if len(test_400ns) !=0:
                ab_points, labels = d_util.IsolationForest_detect(
                    train_arr=train_400ns,
                    test_arr=test_400ns
                )
                ab_points = ab_points.tolist()
                ab_t = test_sts[labels==-1]
                if len(ab_points) > 0:
                    events.append([ab_t[0], callee, caller, '400'])

            # detect business anomaly using rule
            # df_400 = pairs[pairs['status_code']==400]
            # if len(df_400) != 0:
            #     t_400=df_400['start_time'].values.tolist()[0]
            #     events.append([t_400, callee, caller, '400'])

            # df_500 = pairs[pairs['status_code']==500]
            # if len(df_500) != 0:
            #     t_500=df_500['start_time'].values.tolist()[0]
            #     events.append([t_500, callee, caller, '500'])
            

    # sort by timestamp
    sorted_events = sorted(events, key=lambda e:e[0])
    # remove timestamp
    sorted_events = [e[1:] for e in sorted_events]
    return sorted_events


if __name__ == '__main__':
    edges=[('webservice1', 'mobservice1'), ('webservice1', 'mobservice2'), ('webservice2', 'mobservice1'), ('webservice2', 'mobservice2'), ('webservice1', 'redisservice1'), ('webservice1', 'redisservice2'), ('webservice2', 'redisservice1'), ('webservice2', 'redisservice2'), ('mobservice1', 'redisservice1'), ('mobservice1', 'redisservice2'), ('mobservice2', 'redisservice1'), ('mobservice2', 'redisservice2'), ('logservice1', 'dbservice1'), ('logservice1', 'dbservice2'), ('logservice2', 'dbservice1'), ('logservice2', 'dbservice2'), ('logservice1', 'redisservice1'), ('logservice1', 'redisservice2'), ('logservice2', 'redisservice1'), ('logservice2', 'redisservice2'), ('dbservice1', 'redisservice1'), ('dbservice1', 'redisservice2'), ('dbservice2', 'redisservice1'), ('dbservice2', 'redisservice2'), ('logservice1', 'logservice2'), ('logservice2', 'logservice1')]
    # nodes: list = io_util.load('nodes.pkl')
    # edges: tuple = io_util.load('edges.pkl')
    normal_data = io_util.load('normal.pkl')['trace']
    labels = pd.read_csv('gaia.csv')
    labels['st_time']=labels['st_time'].apply(time2stamp)
    
    res = {}
    win_size=30*1000
    normal_res = process_normal(normal_data, win_size, edges)
    for f in os.listdir('pkl'):
        data=io_util.load(f'pkl/{f}')
        for idx in data.keys():
            df = data[idx]['trace']
            events = extract_events(df, normal_res, win_size, edges)
            res[idx] = events
        del data
    # data=io_util.load('gaia.pkl')
    # for idx in data.keys():
    #     df = data[idx]['trace']
    #     events = extract_events(df, nodes, edges)
    #     res[idx] = events

    with open(f"events/trace.json", 'w') as f:
        json.dump(res, f)
    print('Save trace.json successfully!')