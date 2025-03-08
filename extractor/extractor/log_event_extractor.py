import pandas as pd
from tqdm import tqdm

from drain.drain_template_extractor import *


def processing_feature(svc, log, miner):   
    cluster = miner.match(log)
    if cluster is None:
        eventId = -1
    else:
        eventId = cluster.cluster_id
    res = {'service':svc,'id':eventId, 'count':1}
    return res

def extract_log_events(log_df: pd.DataFrame, miner: drain3.TemplateMiner, low_freq_p: float):
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=False)
    err_keywords = ['error', 'fail', 'exception']
    select_events = ['-1']
    for idx, c in enumerate(sorted_clusters):
        if idx < int(low_freq_p * len(sorted_clusters)):
            # low-frequency templates
            select_events.append(c.cluster_id)
            continue
        for keyword in err_keywords:
            if keyword in c.get_template().lower():
                # error log templates
                select_events.append(c.cluster_id)

    log_df.sort_values(by=['timestamp'], ascending=True, inplace=True)
    logs=log_df['message'].values
    svcs=log_df['service'].values

    events_dict = {'service':[], 'id': [], 'count':[]}
    for i,log in tqdm(enumerate(logs)):
        res=processing_feature(svcs[i], log, miner)
        events_dict['service'].append(res['service'])
        events_dict['id'].append(res['id'])
        events_dict['count'].append(res['count'])
    event_df=pd.DataFrame(events_dict)
    event_df = event_df[event_df['id'].isin(select_events)]
    event_gp = event_df.groupby(['id', 'service'])
    events=[[svc, str(event_id)] for (event_id, svc), _ in event_gp]

    return events

