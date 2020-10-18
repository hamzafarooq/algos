import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# using the make_blobs dataset
from sklearn.datasets.samples_generator import make_blobs
X_actual, y = make_blobs(n_samples=1000, centers=5, random_state=101)
# setting the number of training examples
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math



import random

def euclidean(x1,x2):
  return np.sqrt(np.sum((x1-x2)**2))


class Kmeansfromscratch:
  def __init__(self,df,k=5,max_iter=300):
    self.k=k
    self.max_iter=max_iter
    self.df = df

  def fit(self):

    idx = np.random.choice(len(self.df),self.k,replace=False)
    self.centroids = self.df[idx,:]


    P = np.argmin(distance.cdist(self.df, self.centroids, 'euclidean'),axis=1)
    for _ in range(self.max_iter):
      self.centroids = np.vstack([self.df[P==i,:].mean(axis=0) for i in range(self.k)])
      tmp = np.argmin(distance.cdist(self.df, self.centroids, 'euclidean'),axis=1)
      if np.array_equal(P,tmp) :break
      P = tmp

    return P
