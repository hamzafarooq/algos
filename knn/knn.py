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
        self.Y = Y

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x)






