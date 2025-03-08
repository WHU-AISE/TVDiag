import os

import numpy as np
from tqdm import tqdm
from utils import io_util
from utils.time_util import *
import pandas as pd
import time
import random
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

random.seed(12)
np.random.seed(12)

def process_traces(dir):
    def spans_df_left_join(spans_df_ori_1: pd.DataFrame) -> pd.DataFrame:
        """
            加入parent_name属性
        """
        spans_df_temp: pd.DataFrame = spans_df_ori_1
        # 只需要span_id和cmdb_id
        spans_df_ori_1 = spans_df_ori_1.loc[:, ['span_id', 'service_name']]
        # 重命名
        spans_df_ori_1.rename(columns={'service_name': 'parent_name'}, inplace=True)
        start_time = time.time()
        spans_df_temp = spans_df_temp.merge(spans_df_ori_1, left_on='parent_id', right_on='span_id', how='left')
        end_time = time.time()
        process_time = end_time - start_time
        print(fr"用时{process_time}, spans左外连接完成")
        del spans_df_ori_1

        spans_df_temp.rename(columns={'span_id_x': 'span_id'}, inplace=True)
        spans_df_temp.drop(columns=['span_id_y'], inplace=True)
        return spans_df_temp

    def trans2timestamp(df: pd.DataFrame):
        df['start_time'] = df['start_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
        df['end_time'] = df['end_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
        return df

    dfs = []
    for f in os.listdir(dir):
        if f.endswith("2021-07.csv"):
            dfs.append(pd.read_csv(f"{dir}/{f}"))

    trace_df = pd.concat(dfs)
    trace_df = spans_df_left_join(trace_df)
    trace_df = trans2timestamp(trace_df)

    trace_df.to_csv(f"trace.csv")


def process_logs(dir):
    def extract_Date(df: pd.DataFrame):
        df.dropna(axis=0, subset=['message'], inplace=True)
        df['timestamp'] = df['message'].map(lambda m: m.split(',')[0])
        df['timestamp'] = df['timestamp'].apply(lambda x: time2stamp(str(x)))
        return df

    dfs = []
    for f in os.listdir(dir):
        if f.endswith("2021-07.csv"):
            df = pd.read_csv(f"{dir}/{f}")
            df = extract_Date(df)
            dfs.append(df)
    log_df = pd.concat(dfs)
    log_df.to_csv("log.csv")


def extract_traces(trace_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    con1 = trace_df['start_time'] > start_time - 4*window
    con2 = trace_df['start_time'] < start_time
    con3 = trace_df['start_time'] > start_time
    con4 = trace_df['start_time'] < start_time + window
    return trace_df[con1 & con2], trace_df[con3 & con4]


def extract_logs(log_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    con1 = log_df['timestamp'] > start_time - 4*window
    con2 = log_df['timestamp'] < start_time
    con3 = log_df['timestamp'] > start_time
    con4 = log_df['timestamp'] < start_time + window
    return log_df[con1 & con2], log_df[con3 & con4]


def extract_metrics(metric_df: pd.DataFrame, start_time):
    window = 10 * 60 * 1000
    con1 = metric_df['timestamp'] > start_time - 4*window
    con2 = metric_df['timestamp'] < start_time
    con3 = metric_df['timestamp'] > start_time
    con4 = metric_df['timestamp'] < start_time + window
    return metric_df[con1 & con2], metric_df[con3 & con4]


def read_all_metrics():
    svcs = ['dbservice', 'mobservice', 'logservice', 'webservice', 'redisservice']
    pod_names = []
    data = {}
    for svc in svcs:
        pod1 = svc+'1'
        pod2 = svc+'2'
        pod_names.extend([pod1, pod2])
    for f in os.listdir("MicroSS/metric"):
        splits = f.split('_')
        cur_pod, cur_host = splits[0], splits[1]
        if (cur_pod not in pod_names) or ('2021-07-15_2021-07-31' in f):
            continue
        metric_name = '_'.join(splits[2:-2])
        df1 = pd.read_csv(f'MicroSS/metric/{f}')
        next_name = f.replace(
                        "2021-07-01_2021-07-15",
                        "2021-07-15_2021-07-31"
                    )
        df2 = pd.read_csv(f'MicroSS/metric/{next_name}')
        key = cur_pod + '_' + cur_host
        if key not in data.keys():
            data[key] = {}
        data[key][metric_name] = pd.concat([df1, df2])
    return data

if __name__ == '__main__':
    # trace_df = process_traces("trace")
    # log_df = process_logs("business")

    label_df = pd.read_csv("MicroSS/gaia.csv")
    label_df['st_time'] = label_df['st_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    label_df['ed_time'] = label_df['ed_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))

    trace_df = pd.read_csv("MicroSS/trace.csv")
    log_df = pd.read_csv("MicroSS/log.csv")

    pre_data, post_data = {}, {}
    metric_data = read_all_metrics()
    for _, row in label_df.iterrows():
        start_time = time.time()
        idx = row['index']
        pre_data[idx], post_data[idx] = {}, {}
        st_time, ed_time = row['st_time'], row['ed_time']
        pre_trace_df, post_trace_df = extract_traces(trace_df, st_time)
        pre_data[idx]['trace'] = pre_trace_df
        post_data[idx]['trace'] = post_trace_df

        pre_log_df, post_log_df = extract_logs(log_df, st_time)
        pre_data[idx]['log'] = pre_log_df
        post_data[idx]['log'] = post_log_df

        # 并行处理metric
        results = []
        # with mp.Pool(processes=4) as pool:
        #     for f in os.listdir("metric"):
        #         if f.endswith("07-15.csv"):
        #             df1 = pd.read_csv(f"metric/{f}")
        #             next_name = f.replace(
        #                 "2021-07-01_2021-07-15",
        #                 "2021-07-15_2021-07-31"
        #             )
        #             df2 = pd.read_csv(f"metric/{next_name}")
        #             metric_df = pd.concat([df1, df2])

        #             metric_name = f.split("_2021")[0]
        #             metric_df.rename(columns={"value": metric_name}, inplace=True)

        #             result = pool.apply_async(extract_metrics, [metric_df, st_time])
        #             results.append(result)
        #     [result.wait() for result in results]
        pre_metrics, post_metrics = {}, {}
        for pod, metric_dic in metric_data.items():
            pre_metrics[pod], post_metrics[pod] = {}, {}
            for metric_name, metric_df in metric_dic.items():
                pre_metrics[pod][metric_name], post_metrics[pod][metric_name] = extract_metrics(metric_df, st_time)

        pre_data[idx]['metric'] = pre_metrics
        post_data[idx]['metric'] = post_metrics

        end_time = time.time()
        process_time = end_time - start_time
        print(fr"完成{idx}, 用时{process_time}")

    # io_util.save("MicroSS/pre-data.pkl", pre_data)
    io_util.save("MicroSS/post-data-10.pkl", post_data)
