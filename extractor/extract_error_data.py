import os
from utils import io_util
from utils.time_util import *
import pandas as pd
import time
import multiprocessing as mp

def process_traces():
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


    dfs = []
    for f in os.listdir("trace"):
        if f.endswith("2021-07.csv"):
            dfs.append(pd.read_csv(f"trace/{f}"))

    trace_df = pd.concat(dfs)
    trace_df = spans_df_left_join(trace_df)
    trace_df['timestamp']=trace_df['timestamp'].apply(time2stamp)
    trace_df['start_time']=trace_df['start_time'].apply(time2stamp)
    trace_df['end_time']=trace_df['end_time'].apply(time2stamp)
    trace_df['duration']=trace_df['end_time']-trace_df['start_time']

    trace_df.to_csv("trace.csv")


def process_logs():
    def extract_Date(df: pd.DataFrame):
        df.dropna(axis=0, subset=['message'], inplace=True)
        df['timestamp'] = df['message'].map(lambda m: m.split(',')[0])
        df['timestamp'] = df['timestamp'].apply(lambda x: time2stamp(str(x)))
        return df

    dfs = []
    for f in os.listdir("business"):
        if f.endswith("2021-07.csv"):
            df = pd.read_csv(f"business/{f}")
            df = extract_Date(df)
            dfs.append(df)
    log_df = pd.concat(dfs)
    log_df.to_csv("log.csv")


# def process_metrics():
#     metric_dict = {}
#     for f in os.listdir("metric"):
#         metric_name = f.split("_2021")[0]
#         metric_df = pd.read_csv(f"metric/{f}")
#         # metric_df.set_index('timestamp', inplace=True)
#         metric_df.rename(columns={"value": metric_name}, inplace=True)
#         if metric_name in metric_dict.keys():
#             metric_dict[metric_name] = pd.concat([metric_dict[metric_name], metric_df])
#             # metric_dict[metric_name].sort_index(inplace=True)
#             metric_dict[metric_name].sort_values(by=['timestamp'], ascending=True)
#             metric_dict[metric_name].drop_duplicates(subset=['timestamp'], inplace=True)
#             metric_dict[metric_name].set_index('timestamp', inplace=True)
#         else:
#             metric_dict[metric_name] = metric_df
#
#     dfs = list(metric_dict.values())
#     i, id, n = 0, 0, int(len(dfs) / 10)
#     # 分割合并
#     while i < len(dfs):
#         df = pd.concat(dfs[i:i+n], axis=1)
#         df.to_csv(f'metric{id}.csv')
#         i+=n
#         id+=1




def extract_traces(trace_df: pd.DataFrame, start_time, end_time):
    window=60*1000
    con1 = trace_df['timestamp'] > start_time
    con2 = trace_df['timestamp'] < end_time+1*window
    return trace_df[con1 & con2]


def extract_logs(log_df: pd.DataFrame, start_time, end_time):
    con1 = log_df['timestamp'] > start_time
    con2 = log_df['timestamp'] < end_time
    return log_df[con1 & con2]


def extract_metrics(metric_df: pd.DataFrame, start_time, end_time):
    window=1*60*1000
    con1 = metric_df['timestamp'] > start_time-40*window
    con2 = metric_df['timestamp'] < end_time+10*window
    return metric_df[con1 & con2]


if __name__ == '__main__':
    # trace_df = process_traces()
    # log_df = process_logs()

    label_df = pd.read_csv("gaia.csv")
    label_df['st_time'] = label_df['st_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    label_df['ed_time'] = label_df['ed_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    idxs = label_df.index.values.tolist()

    trace_df = pd.read_csv("trace.csv")
    # log_df = pd.read_csv("log.csv")

    # reduce the IO count
    # metric_fs = {}
    # for f in os.listdir('metric'):
    #     if f.startswith("zookeeper") or f.startswith("system"):
    #         continue
    #     metric_fs[f] = pd.read_csv(f'metric/{f}')

    data = io_util.load('gaia.pkl')
    for idx, row in label_df.iterrows():
        start_time = time.time()
        # data[idx] = {}
        st_time, ed_time = row['st_time'], row['ed_time']
        tmp_trace_df = extract_traces(trace_df, st_time, ed_time)
        data[idx]['trace'] = tmp_trace_df

        # tmp_log_df = extract_logs(log_df, st_time, ed_time)
        # data[idx]['log'] = tmp_log_df

        # Parallel Processing
        # results = []
        # with mp.Pool() as pool:
        #     for f_name, metric_f in metric_fs.items():
        #         metric_name = f_name.split("_2021")[0]
        #         metric_f.rename(columns={"value": metric_name}, inplace=True)
        #         result = pool.apply_async(extract_metrics, [metric_f, st_time, ed_time])
        #         results.append(result)
        #     [result.wait() for result in results]

        #     data[idx]['metric'] = [res.get() for res in results if not res.get().empty]

        end_time = time.time()
        process_time = end_time - start_time
        print(fr"完成{idx}, 用时{process_time}")

    io_util.save("gaia.pkl", data)
