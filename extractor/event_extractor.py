from collections import defaultdict
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

from extractor.metric_event_extractor import extract_metric_events
from extractor.trace_event_extractor import extract_trace_events
from extractor.log_event_extractor import extract_log_events
from utils import io_util


data: dict = io_util.load('MicroSS/post-data-10.pkl')
# 将第一列设置为索引
label_df = pd.read_csv('MicroSS/gaia.csv', index_col=0)

metric_detectors = io_util.load('MicroSS/detector/metric-detector-strict-host.pkl')
trace_detectors = io_util.load('MicroSS/detector/trace-detector.pkl')


metric_events_dic = defaultdict(list)
trace_events_dic = defaultdict(list)
log_events_dic = defaultdict(list)
metric_costs, trace_costs, log_costs = [], [], []

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = data[idx]
    # extract metric events
    st = time.time()
    metric_events = []
    for pod_host, kpi_dic in chunk['metric'].items():
        kpi_events = extract_metric_events(pod_host, kpi_dic, metric_detectors[pod_host])
        metric_events.extend(kpi_events)
    metric_costs.append(time.time()-st)
    metric_events_dic[idx]=metric_events
    # extract trace events
    st = time.time()
    trace_events = extract_trace_events(chunk['trace'], trace_detectors)
    trace_events_dic[idx] = trace_events
    trace_costs.append(time.time()-st)
    # extract log events
    st = time.time()
    miner = io_util.load('./drain/gaia-drain.pkl')
    log_df = chunk['log']
    log_events = extract_log_events(log_df, miner, 0.5)
    log_events_dic[idx] = log_events
    log_costs.append(time.time()-st)

metric_time = np.mean(metric_costs)
trace_time = np.mean(trace_costs)
log_time = np.mean(log_costs)
print(f'the time cost of extract metric events is {metric_time}')
print(f'the time cost of extract trace events is {trace_time}')
print(f'the time cost of extract log events is {log_time}')
#the time cost of extract metric events is 0.18307018280029297
# the time cost of extract trace events is 0.23339865726162023
# the time cost of extract log events is 0.6638196256618483

io_util.save_json('events/log/log.json', log_events_dic)
io_util.save_json('events/metric/metric.json', metric_events_dic)
io_util.save_json('events/trace/trace.json', trace_events_dic)