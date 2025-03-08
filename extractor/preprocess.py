# import pandas as pd
# from collections import defaultdict
# from tqdm import tqdm
# from utils import io_util

# labels = pd.read_csv('MicroSS/gaia.csv')
# failure_pre_data: dict = io_util.load('MicroSS/pre-data.pkl')


# normal_metrics = {}
# normal_traces = defaultdict(list)

# for idx, row in tqdm(labels.iterrows(), total=labels.shape[0]):
#     if row['data_type'] == 'test':
#         continue
#     index = row['index']
#     chunk = failure_pre_data[index]
#     for pod, kpi_dic in chunk['metric'].items():
#         if pod not in normal_metrics.keys():
#             normal_metrics[pod] = defaultdict(list)
#         for kpi, kpi_df in kpi_dic.items():
#             normal_metrics[pod][kpi].append(kpi_df)
            
    # trace_df = chunk['trace']
    # trace_df['operation'] = trace_df['url'].str.split('?').str[0]
    # trace_gp = trace_df.groupby(['parent_name', 'service_name', 'operation'])
    # for (src, dst, op), call_df in trace_gp:
    #     name = src + '-' + dst + '-' + op
    #     normal_traces[name].append(call_df)

# for pod in normal_metrics.keys():
#     for kpi, kpi_dfs in normal_metrics[pod].items():
#         normal_metrics[pod][kpi] = pd.concat(kpi_dfs)

# io_util.save('MicroSS/detector/normal_traces.pkl', normal_traces)
# io_util.save('MicroSS/detector/normal_metrics.pkl', normal_metrics)

############################################################################

import numpy as np
from sklearn.ensemble import IsolationForest
from extractor.trace_event_extractor import slide_window
from utils import io_util
import time



normal_traces = io_util.load('MicroSS/detector/normal_traces.pkl')
normal_metrics = io_util.load('MicroSS/detector/normal_metrics.pkl')

metric_detectors = {}
for pod in normal_metrics.keys():
    metric_detectors[pod] = {}
    for kpi, dfs in normal_metrics[pod].items():
        metric_detectors[pod][kpi] = [
            normal_metrics[pod][kpi]['value'].mean(), 
            normal_metrics[pod][kpi]['value'].std()
        ]
st = time.time()
trace_detectors = {}
for name, call_dfs in normal_traces.items():
    trace_detectors[name] = {
        'dur_detector': IsolationForest(random_state=0, n_estimators=5),
        '500_detector': IsolationForest(random_state=0, n_estimators=5),
        '400_detector': IsolationForest(random_state=0, n_estimators=5)
    }
    train_ds, train_500_ep, train_400_ep = [], [], []
    for call_df in call_dfs:
        _, durs, err_500_ps, err_400_ps = slide_window(call_df, 30 * 1000)
        train_ds.extend(durs)
        train_500_ep.extend(err_500_ps)
        train_400_ep.extend(err_400_ps)
    if len(train_ds) == 0:
        continue
    dur_clf, err_500_clf, err_400_clf = trace_detectors[name]['dur_detector'], trace_detectors[name]['500_detector'], trace_detectors[name]['400_detector']
    dur_clf.fit(np.array(train_ds).reshape(-1,1))
    err_500_clf.fit(np.array(err_500_ps).reshape(-1,1))
    err_400_clf.fit(np.array(err_400_ps).reshape(-1,1))
    trace_detectors[name]['dur_detector']=dur_clf
    trace_detectors[name]['500_detector']=err_500_clf
    trace_detectors[name]['400_detector']=err_400_clf

ed = time.time()
io_util.save('MicroSS/detector/trace-detector.pkl', trace_detectors)
io_util.save('MicroSS/detector/metric-detector-strict-host.pkl', metric_detectors)

print(ed-st)