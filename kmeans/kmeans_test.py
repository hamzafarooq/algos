from sklearn.datasets.samples_generator import make_blobs
X_actual, y = make_blobs(n_samples=1000, centers=5, random_state=101)

from kmeans import Kmeansfromscratch

import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math



X=X_actual
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
clf=Kmeansfromscratch(X,5,600)

P=clf.fit()

X = sc.inverse_transform(X)
plt.figure(figsize=(15,10))
plt.scatter(X[:,0],X[:,1],c=P)
plt.show()
