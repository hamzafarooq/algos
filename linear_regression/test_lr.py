import numpy as np
import pandas as  pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from linear_regression import LinearRegression


X, y = make_regression(n_samples=300, n_features=1, n_informative=1, noise=6, bias=30, random_state=200)
m = 200


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#fig = plt.figure(figsize=(10,15))

#plt.scatter(X[:,0],y, color="b" , marker="o",s = 30)
#plt.show()

#print(X_train.shape)
#print(y_train.shape)



reg = LinearRegression(0.025,1000)
reg.fit(X_train,y_train)

predicted = reg.predict(X_test)

def mse(y_hat,y):
    return np.mean((y_hat - y)**2)


print(mse(predicted,y_test))

fig = plt.figure(figsize=(10,15))

plt.scatter(X_test[:,0],y_test, color="b" , marker="o",s = 30)
plt.scatter(X_test[:,0],predicted, color="r" , marker="x",s = 30)
plt.show()
