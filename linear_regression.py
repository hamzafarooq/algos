import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, n_iter = 400):
        self.lr = lr
        self.n_iter = 400
        self.weights = None
        self.bias = None

    def fit(self,X,y):

        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_predicted = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T,(y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self,X):
        y_predicted = np.dot(X,self.weights) + self.bias
        return y_predicted
