import json
import os

import pandas as pd
import utils.io_util as io_util
import utils.detect_util as d_util
from utils.time_util import coast_time, time2stamp

def findSubStr_clark(sourceStr, str, i):
    count = 0
    rs = 0
    for c in sourceStr:
        if c == str:
            count += 1
        if count == i:
            return rs
        rs += 1
    return -1

@coast_time
def extract_events(dfs: list, st_time):
    """extract events using 3-sigma from 
        different metric dataframe

    Args:
        dfs (list): metric dataframes

    Returns:
        events: extracted abnormal events
    """
    events = []
    # kpis=['system_core_iowait_pct', 'system_core_system_pct', 'zookeeper_server_outstanding', 'docker_cpu_kernel_pct', 'docker_memory_stats_pgpgin', 'system_diskio_iostat_write_await', 'docker_memory_stats_pgpgout', 'system_core_id', 'redis_info_slowlog_count', 'zookeeper_mntr_outstanding_requests', 'docker_cpu_core_0_pct', 'redis_info_persistence_aof_fsync_pending', 'system_diskio_iostat_queue_avg_size', 'redis_info_persistence_aof_copy_on_write_last_size', 'system_cpu_nice_pct', 'docker_cpu_core_1_pct', 'redis_info_memory_used_dataset', 'system_diskio_iostat_read_per_sec_bytes', 'system_diskio_iostat_write_request_per_sec', 'system_cpu_iowait_pct', 'docker_diskio_write_rate', 'redis_info_persistence_aof_rewrite_last_time_sec', 'redis_info_stats_latest_fork_usec', 'system_process_cpu_total_pct', 'redis_info_persistence_rdb_last_save_changes_since', 'redis_keyspace_avg_ttl', 'docker_cpu_core_9_pct', 'system_diskio_iostat_read_await', 'docker_cpu_core_2_pct', 'docker_cpu_core_13_pct', 'system_diskio_iostat_write_per_sec_bytes', 'redis_info_persistence_aof_size_current', 'docker_memory_stats_dirty', 'system_load_1', 'docker_cpu_total_pct', 'system_core_user_pct', 'system_cpu_system_pct', 'system_core_idle_pct', 'system_diskio_iostat_busy', 'system_diskio_iostat_await', 'docker_network_out_bytes', 'system_core_nice_pct', 'system_core_softirq_pct', 'docker_cpu_core_11_pct', 'redis_info_persistence_aof_size_base', 'redis_info_persistence_rdb_copy_on_write_last_size', 'system_diskio_iostat_write_request_merges_per_sec', 'docker_cpu_core_3_pct', 'docker_cpu_core_8_pct', 'redis_keyspace_keys', 'docker_cpu_system_pct', 'system_diskio_iostat_read_request_merges_per_sec', 'docker_cpu_core_15_pct', 'docker_cpu_core_5_pct', 'docker_cpu_core_6_pct', 'docker_cpu_core_7_pct', 'system_diskio_iostat_request_avg_size', 'docker_cpu_user_pct', 'system_process_memory_share', 'docker_diskio_read_rate', 'docker_cpu_core_4_pct', 'redis_info_stats_instantaneous_input_kbps', 'system_process_summary_running', 'redis_info_clients_connected', 'redis_info_stats_instantaneous_output_kbps', 'system_cpu_softirq_pct', 'docker_cpu_core_10_pct', 'docker_memory_usage_pct', 'redis_info_persistence_rdb_bgsave_last_time_sec', 'docker_cpu_core_12_pct', 'redis_info_persistence_aof_buffer_size', 'docker_memory_stats_cache', 'system_diskio_io_ops', 'redis_info_memory_used_rss', 'docker_cpu_core_14_pct', 'docker_network_in_bytes', 'redis_info_clients_max_input_buffer', 'zookeeper_server_count', 'system_memory_actual_used_pct', 'system_diskio_iostat_read_request_per_sec', 'docker_memory_stats_active_anon']
    for df in dfs:      
        full_name = [col for col in df.columns if col != 'timestamp'][0]
        split_idx = findSubStr_clark(full_name, '_', 2) # the second _
        svc_host = full_name[:split_idx]
        metric_name = full_name[split_idx+1:]
        svc, host = svc_host.split('_')[0], svc_host.split('_')[1]

        # if metric_name not in kpis:
        #     continue

        df.fillna(0, inplace=True)
        df.sort_values(by=['timestamp'], inplace=True, ascending=True)

        train_df=df[df['timestamp']<st_time]
        test_df=df[df['timestamp']>st_time]
        
        times = test_df['timestamp'].values
        if len(test_df)==0 or len(train_df)==0:
            continue
        # detect anomaly using 3-sigma
        # cur_events, labels = d_util.IsolationForest_detect(
        #     train_arr=train_df[full_name].values,
        #     test_arr=test_df[full_name].values
        # )

        cur_events, labels = d_util.k_sigma(
            train_arr=train_df[full_name].values,
            test_arr=test_df[full_name].values,
            k=3,
        )
        # cur_events, labels = d_util.IsolationForest_detect(
        #     train_arr=train_df[full_name].values,
        #     test_arr=test_df[full_name].values,
        # )
        
        # cur_events, labels = d_util.SVM_detect(
        #     train_arr=train_df[full_name].values,
        #     test_arr=test_df[full_name].values,
        # )

        cur_events = cur_events.tolist()
        if len(cur_events) == 0:
            continue
        ab_t = times[labels==-1]
        
        # # construct structual events
        # cur_events = [[ab_t[i], host, svc, metric_name] for i, e in enumerate(cur_events)]
        # events.extend(cur_events)
        events.append([ab_t[0], svc, host, metric_name])
    
    # sort by timestamp
    sorted_events = sorted(events, key=lambda e:e[0])
    # remove timestamp
    sorted_events = [e[1:] for e in sorted_events]
    return sorted_events


if __name__ == '__main__':
    res = {}
    labels = pd.read_csv('gaia.csv')
    labels['st_time']=labels['st_time'].apply(time2stamp)

    for f in os.listdir('pkl'):
        data=io_util.load(f'pkl/{f}')
        for idx in data.keys():
            m_dfs = data[idx]['metric']
            events = extract_events(m_dfs, labels.loc[idx, 'st_time'])
            res[idx] = events
        del data

    with open(f"events/metric.json", 'w') as f:
        json.dump(res, f)
    print('Save metric.json successfully!')