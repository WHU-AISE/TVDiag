from collections import defaultdict
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from extractor.metric_event_extractor import extract_metric_events
from extractor.trace_event_extractor import extract_trace_events
from extractor.log_event_extractor import extract_log_events
from utils import io_util


failure_post_data: dict = io_util.load('MicroSS/post-data.pkl')
# 将第一列设置为索引
label_df = pd.read_csv('MicroSS/gaia.csv', index_col=0)

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = failure_post_data[idx]
    # 对每个label提取依赖关系（服务之间是调用关系，同一个节点的服务为双向边）
    trace_df = chunk['trace']
    svcs = []
    influences = []

    # 捕获服务与节点的对应关系
    node2svcs = defaultdict(list)
    for f in os.listdir('./MicroSS/metric'):
        splits = f.split('_')
        svc, host = splits[0], splits[1]
        if svc in ['system', 'redis', 'zookeeper']:
            continue
        if svc not in node2svcs[host]:
            node2svcs[host].append(svc)
        
    for node, pods in node2svcs.items():
        # print(node)
        # print(pods)
        # print('============================')
        svcs.extend(pods)
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                influences.append([pods[i], pods[j]]) 
                influences.append([pods[j], pods[i]])
    svcs = list(set(svcs))
    svcs.sort()
    # print(svcs)
    # print(len(svcs))

    # 捕获服务调用关系
    edges = []
    edge_columns = ['service_name', 'parent_name']
    calls = trace_df.dropna(subset=['parent_name']).drop_duplicates(subset=edge_columns)[edge_columns].values.tolist()
    calls.extend(influences)
    calls = pd.DataFrame(calls).drop_duplicates().reset_index(drop=True).values.tolist() # 去重
    # print(calls)
    # print(len(calls))
    for call in calls:
        source, target = call[1], call[0]
        source_idx, target_idx = svcs.index(source), svcs.index(target)
        edges.append([source_idx, target_idx])
    chunk['nodes'] = svcs
    chunk['edges'] = edges

############################################################################################################################
from tqdm import tqdm
edges = {}
nodes = {}
for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = failure_post_data[idx]
    edges[idx] = chunk['edges']
    nodes[idx] = chunk['nodes']
io_util.save_json('MicroSS/nodes.json', nodes)
io_util.save_json('MicroSS/edges.json', edges)