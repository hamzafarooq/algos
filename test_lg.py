import numpy as np
import pandas as  pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1234)



reg = LogisticRegression(lr=0.0001,n_iter=1000)
reg.fit(X_train,y_train)

predicted = reg.predict(X_test)

def accuracy(y_hat,y):
    return (np.sum(y_hat == y)/len(y))


print(f'The accuracy is {accuracy(predicted,y_test)}')
