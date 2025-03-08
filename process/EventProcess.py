import os
from torch.utils.data import DataLoader
from config.exp_config import Config
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
import json
import pandas as pd
import numpy as np
from process.events.fasttext_w2v import FastTextEncoder
from core.aug import *
import time

class EventProcess():

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.dataset = config.dataset

    def process(self, reconstruct=False):
        self.data_path = f"data/{self.dataset}"
        label_path = f"data/{self.dataset}/label.csv"
        metric_path = f"data/{self.dataset}/raw/metrics.json"
        trace_path = f"data/{self.dataset}/raw/traces.json"
        log_path = f"data/{self.dataset}/raw/logs.json"
        edge_path = f"data/{self.dataset}/raw/edges.json"
        node_path = f"data/{self.dataset}/raw/nodes.json"

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)
        self.labels['index'] = self.labels['index'].astype(str)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)
        with open(edge_path, 'r', encoding='utf8') as fp:
            self.edges = json.load(fp)
        with open(node_path, 'r', encoding='utf8') as fp:
            self.nodes = json.load(fp)

        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist()

        if reconstruct:
            self.build_embedding()

        return self.build_dataset()

    def build_embedding(self):
        self.logger.info(f"Build embedding for raw events")
        # metric event: (instance, metric_name, abnormal type)
        # trace event: (src, dst, op, error_type)
        # log event: (instance, eventId)

        data_map = {'metric': self.metrics, 'trace': self.traces, 'log': self.logs}
        # data_map = {'trace': self.traces}
        
        for key, data in data_map.items():
            all_nodes = list({item for sublist in self.nodes.values() for item in sublist})
            encoder = FastTextEncoder(key, all_nodes, self.types, embedding_dim=self.config.alert_embedding_dim, epochs=5)

            train_idxs = self.labels[self.labels['data_type']=='train']['index'].values.tolist()
            train_ins_labels = self.labels[self.labels['data_type']=='train']['instance'].values.tolist()
            train_type_labels = self.labels[self.labels['data_type']=='train']['anomaly_type'].values.tolist()
            docs = []
            labels = []
            for i, idx in enumerate(train_idxs):
                nodes = self.nodes[str(idx)]
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    docs.append(doc)
                    if node == train_ins_labels[i]:
                        labels.append(f'__label__{node}{self.types.index(train_type_labels[i])}')
                    else:
                        labels.append(f'__label__{node}0')
            encoder.fit(docs, labels)

            # build embedding
            embs = {}
            for idx in self.labels['index']:
                # group by instance
                graph_embs = []
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    
                    emb = encoder.get_sentence_embedding(doc)
                    graph_embs.append(emb)
                embs[idx]=graph_embs
            tmp_pth = f'data/{self.dataset}/tmp/'
            if not os.path.isdir(tmp_pth):
                os.system(f"mkdir -p {tmp_pth}")
            io_util.save_pkl(f"{tmp_pth}/{key}.pkl", embs)


    def build_dataset(self):
        self.logger.info(f"Build dataset for training")
        metric_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/metric.pkl")
        trace_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/trace.pkl")
        log_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/log.pkl")

        label_dict = {}
        all_nodes = list({item for sublist in self.nodes.values() for item in sublist})
        all_nodes.sort()
        label_dict['instance'], node2idx, idx2root = self.get_root_labels(all_nodes)
        label_dict['anomaly_type'], ft2idx, idx2ft = self.get_type_labels(self.labels['anomaly_type'].values.tolist())
        
        train_data, test_data = MultiModalDataSet(), MultiModalDataSet()
        for _, row in self.labels.iterrows():
            index = str(row['index'])
            data_type = row['data_type']
            # data
            metric_Xs, trace_Xs, log_Xs = metric_embs[index], trace_embs[index], log_embs[index]
            # labels
            global_root_id = node2idx[row['instance']] # global root cause label (guide the TO)
            failure_type_id = ft2idx[row['anomaly_type']]
            # topo
            nodes = self.nodes[index]
            edges = self.edges[index]
            
            if data_type == 'train':
                train_data.add_data(
                    metric_Xs=metric_Xs, 
                    trace_Xs=trace_Xs, 
                    log_Xs=log_Xs, 
                    global_root_id=global_root_id, 
                    failure_type_id=failure_type_id, 
                    local_root=row['instance'], 
                    nodes=nodes, 
                    edges=edges)
            else:
                test_data.add_data(
                    metric_Xs=metric_Xs, 
                    trace_Xs=trace_Xs, 
                    log_Xs=log_Xs, 
                    global_root_id=global_root_id, 
                    failure_type_id=failure_type_id, 
                    local_root=row['instance'], 
                    nodes=nodes, 
                    edges=edges)
        
        aug_data = []
        if self.config.aug_times > 0:
            for time in range(self.config.aug_times):
                for (graph, labels) in train_data:
                    root = graph.ndata['root'].tolist().index(1)
                    aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)
                    aug_data.append((aug_graph, labels))
        return train_data, aug_data, test_data

    def get_root_labels(self, nodes):
        labels2idx = {node: idx for idx, node in enumerate(nodes)}
        idx2label = {idx: label for idx, label in enumerate(nodes)}
        labels = np.array(self.labels['instance'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label

    def get_type_labels(self, types):
        meta_labels = sorted(list(set(types)))
        labels2idx = {label: idx for idx, label in enumerate(meta_labels)}
        idx2label = {idx: label for idx, label in enumerate(meta_labels)}
        labels = np.array(self.labels['anomaly_type'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label