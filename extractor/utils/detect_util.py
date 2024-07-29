import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import svm

def DBSCAN_detect(train_arr, test_arr, eps=3, min_samples=2):
    if len(test_arr) == 0:
        return np.array([]), np.array([0]*len(test_arr))
    test_data = test_arr.reshape(-1,1)
    train_data = train_arr.reshape(-1,1)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(train_data)
    labels=model.fit_predict(test_data)
    # model.fit_predict(test_data)
    # get labels (-1 denotes anomaly)
    ab_points = test_arr[labels==-1]

    return ab_points, labels

def k_sigma(train_arr, test_arr: np.array, k=3):
    mean = np.mean(train_arr)
    std = np.std(train_arr)
    up, lb=mean+k*std, mean-k*std
    ab_points=test_arr[(test_arr>up)|(test_arr<lb)]
    labels=np.array([0]*len(test_arr))
    ab_idxs=np.where((test_arr>up)|(test_arr<lb))
    labels[ab_idxs]=-1
    return ab_points, labels


def IsolationForest_detect(train_arr, test_arr):
    clf = IsolationForest(random_state=0, n_estimators=5)
    clf.fit(train_arr.reshape(-1,1))
    labels = clf.predict(test_arr.reshape(-1,1))
    ab_points = test_arr[labels==-1]
    return ab_points, labels

def SVM_detect(train_arr, test_arr):
    clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    clf.fit(train_arr.reshape(-1,1))
    labels = clf.predict(test_arr.reshape(-1,1))
    ab_points = test_arr[labels==-1]
    return ab_points, labels
    