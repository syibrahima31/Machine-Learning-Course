# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:46:01 2021

@author: sy ibrahima 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression

# create datasets 
X, y = make_regression(n_samples =1000, n_features=1 , noise = 100)

plt.figure()
plt.scatter(X, y)

class LinearRegression:
    def __init__(self, n_iters = 100 , lr =0.01):
        self.n_iters = n_iters 
        self.lr = lr 
    
    def fit(self, X_train , y_train):
        self.X = X_train 
        self.y = y_train.reshape(-1,1)
        
        n_samples , n_features = self.X.shape 
        
        # redifine self.X 
        vec_ones = np.ones( n_samples).reshape(-1,1)
        
        # concatenate 
        self. X = np.hstack([vec_ones, self.X])
        
        # define model 
        self.beta = np.random.randn(n_features +1 ).reshape(-1,1)
        
        Y_pred = self.X.dot(self.beta)
        
        # Gradient 
        gradient = (1/ n_samples)* self.X.T.dot(Y_pred - self.y)
        
        # descent gradient 
        for i in range(self.n_iters):
            self.beta = self.beta - self.lr * gradient 
            Y_pred = self.X.dot(self.beta)
        
        
    def predict(self, X):
        
        
        n_samples , n_features = X.shape 
            
        # redifine self.X 
        vec_ones = np.ones( n_samples).reshape(-1,1)
            
        # concatenate 
        X = np.hstack([vec_ones, X])
        return X.dot(self.beta)
            
                


model= LinearRegression()
model.fit(X, y)
prediction =  model.predict(X)



plt.figure()
plt.scatter(X, y)
plt.plot(X, prediction, "r")
