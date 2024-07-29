import os
import random
from typing import List
import fasttext
import numpy as np

"""
    adapted from:
    
    Zhang S, Jin P, Lin Z, et al. Robust Failure Diagnosis of Microservice System through Multimodal Data[J]. 
    https://arxiv.org/abs/2302.10512
"""

class FastTextEncoder:
    def __init__(self, modality, nodes, types, embedding_dim=100, epochs=5, random_state=1):
        self.dim = embedding_dim
        self.epochs = epochs
        self.random_state=random_state

        self.nodes = nodes
        self.types = types
        self.modality = modality

    def build_datasets(self, data_set, labels, model):
        final_data = data_set.copy()
        
        for type in self.types:
            for node in self.nodes:
                idxs = [i for i, label in enumerate(labels)
                    if label.split('__label__')[-1] == str(self.nodes.index(node)) + str(
                        self.types.index(type))]
                sample_count = len(idxs)
                if sample_count == 0:
                    continue
                anomaly_texts = [data_set[idx] for idx in idxs]
                loop = 0
                sample_num = 1000
                while sample_count < sample_num:
                    loop += 1
                    if loop >= 10 * sample_num:
                        break
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < 1:
                        continue
                    edit_event_ids = random.sample(range(len(chosen_text_splits)), 1)
                    for event_id in edit_event_ids:
                        nearest_event = model.get_nearest_neighbors(chosen_text_splits[event_id])[0][-1]
                        chosen_text_splits[event_id] = nearest_event
                    final_data.append(
                        ' '.join(
                            chosen_text_splits) + f'\t__label__{self.nodes.index(node)}{self.types.index(type)}\n')
                    sample_count += 1
        
        return final_data


    def save_to_txt(self, data_set, filename):
        os.makedirs('./tmp', exist_ok=True)
        path = f'./tmp/{filename}'
        with open(path, 'w') as f:
            for text in data_set:
                f.write(text) 

        return path


    def fit(self, data_set: List[List[str]], labels):
        data_set = [' '.join(events) for events in data_set]
        data_set = [f'{text}\t{labels[i]}\n' for i, text in enumerate(data_set)]
        train_pth = self.save_to_txt(data_set, f'{self.modality}-train.txt')
        model = fasttext.train_supervised(
            train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )

        # aug
        aug_data_set = self.build_datasets(data_set, labels, model)
        aug_train_pth = self.save_to_txt(aug_data_set, f'{self.modality}-train-aug.txt')
        model = fasttext.train_supervised(
            aug_train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )

        # event embedding
        self.event_dic = {}
        for e in model.words:
            self.event_dic[e] = model[e]
        

    def get_sentence_embedding(self, text: List[str]) -> List[float]:
        text = ' '.join(text)
        
        # senetence embedding
        length = len(self.event_dic[list(self.event_dic.keys())[0]])
        sen_emb = np.array([0] * length, 'float32')
        if text != '':
            words = list(set(text.split(' ')))
            for word in words:
                if word in self.event_dic:
                    sen_emb = sen_emb + np.array(self.event_dic[word])

        return sen_emb