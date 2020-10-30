import numpy as np
import pandas as pd 
from collections import Counter

def euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class knn:
    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]






