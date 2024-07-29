import json
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import pandas as pd
import os.path as osp

from drain.extract_log_templates import *
import utils.io_util as io_util
from utils.time_util import coast_time

def init_parser(logs: list):
    """ Extract templates of logs
        Transform the logs into embeddings

    Args:
        L_data : The raw log data
    """
    drain_parser = extract_templates(
        log_list=logs,
        save_pth='drain/drain.pkl'
    )

    # save templates and ID
    sorted_clusters = sorted(drain_parser.drain.clusters, key=lambda it: it.size, reverse=True)
    uq_tmps, uq_IDs, sizes = [], [], []
    for cluster in sorted_clusters:
        uq_tmps.append(cluster.get_template())
        uq_IDs.append(cluster.cluster_id)
        sizes.append(cluster.size)
    template_df = pd.DataFrame(data={"id": uq_IDs, "template": uq_tmps, 'count': sizes})
    template_df.to_csv('drain/statistics.csv', index=False)


def processing_feature(svc, log, miner):   
    cluster = miner.match(log)
    if cluster is None:
        eventId = -1
    else:
        eventId = cluster.cluster_id
    res = {'service':svc,'id':eventId, 'count':1}
    return res


@coast_time
def extract_events(log_df: pd.DataFrame, miner: drain3.TemplateMiner, count_dic: dict, k: int):
    log_num = len(log_df)
    print(log_num)
    # templates_total_num = sum(count_dic.values())
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=False)

    select_events = []
    e_num=0
    for c in sorted_clusters:
        if 'ERROR' in c.get_template():
            select_events.append(c.cluster_id)
        elif e_num < k:
            select_events.append(c.cluster_id)
            e_num+=1
        else:
            break

    log_df.sort_values(by=['timestamp'], ascending=True, inplace=True)
    # log_df = log_df.sample(frac=0.01)
    logs=log_df['message'].values
    svcs=log_df['service'].values

    # event_df = pd.DataFrame(
    #         Parallel(n_jobs=mp.cpu_count(), 
    #                 backend="multiprocessing")
    #         (delayed(processing_feature)(svcs[i], log, miner) for i,log in tqdm(enumerate(logs))))
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

    # TF-IDF score
    # for (event_id, svc), df in event_gp:
    #     TF = len(df)/log_num
    #     IDF = np.log(templates_total_num/count_dic[event_id])
    #     TF_IDF = round(TF*IDF,3)
    #     events.append([svc, str(event_id)])

    return events


if __name__ == '__main__':
    labels = pd.read_csv('gaia.csv')
    pods = io_util.load('nodes.pkl')

    # init drain using train dataset
    # train_logs = []
    # train_idxs = labels[labels['data_type']=='train']['index'].values.tolist()
    # for f in os.listdir('pkl'):
    #     data=io_util.load(f'pkl/{f}')
    #     for idx in data.keys():
    #         if idx in train_idxs:
    #             log_df = data[idx]['log']
    #             train_logs.extend(log_df['message'].values.tolist())
    #     del data
    # init_parser(train_logs)

    count_df = pd.read_csv('drain/statistics.csv')
    count_dic=dict(zip(count_df['id'], count_df['count']))
    parser = io_util.load('drain/drain.pkl')
    svc_map = dict.fromkeys(pods, '0')
    k=20
    res = {}

    for f in os.listdir('pkl'):
        print(f"processe the {f}")
        data=io_util.load(f'pkl/{f}')
        for idx in data.keys():
            log_df = data[idx]['log']
            events = extract_events(log_df, parser, count_dic, k)
            res[idx] = events
        del data

    with open(f"events/log.json", 'w') as f:
        json.dump(res, f)
    print('Save log.json successfully!')
