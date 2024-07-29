import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

def min_max_scaler(data):
    # (x - min) / (max - min)
    scaler = MinMaxScaler(feature_range=(0,1))
    return scaler.fit_transform(data)

def quantile_scaler(data):
    scaler = QuantileTransformer()
    return scaler.fit_transform(data)

def standard_scaler(data):
    # (x-mean)/std
    scaler = StandardScaler()
    return scaler.fit_transform(data)