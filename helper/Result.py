import numpy as np
import pandas as pd

class Result:
    def __init__(self):
        pass

    def set_performance(self, rcl_results: dict, fti_results: dict):
        self.hr_1 = rcl_results['HR@1']
        self.hr_2 = rcl_results['HR@2']
        self.hr_3 = rcl_results['HR@3']
        self.hr_4 = rcl_results['HR@4']
        self.hr_5 = rcl_results['HR@5']
        self.mrr_3 = rcl_results['MRR@3']
        self.avg_3 = np.mean([rcl_results['HR@1'], rcl_results['HR@2'], rcl_results['HR@3']])

        self.pre = fti_results['pre']
        self.rec = fti_results['rec']
        self.f1 = fti_results['f1']

    def set_inference_efficiency(self, inference_times: list):
        self.avg_inference_time = np.mean(inference_times)
        self.total_inference_time = np.sum(inference_times)
    
    def set_train_efficiency(self, train_times: list):
        self.avg_train_time = np.mean(train_times)
        self.total_train_time = np.sum(train_times)

    def export_df(self, name):
        df = pd.DataFrame(data={
            'name': [name],
            'HR@1': [self.hr_1],
            'HR@2': [self.hr_2],
            'HR@3': [self.hr_3],
            'HR@4': [self.hr_4],
            'HR@5': [self.hr_5],
            'MRR@3': [self.mrr_3],
            'avg@3': [self.avg_3],
            'pre': [self.pre],
            'rec': [self.rec],
            'f1':   [self.f1],
            'train_time': [self.total_train_time],
            'avg_train_time': [self.avg_train_time],
            'inference_time': [self.total_inference_time],
            'avg_inference_time': [self.avg_inference_time]
        })
        return df