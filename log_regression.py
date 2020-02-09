# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from sklearn.model_selection import train_test_split

class log_regression:
    

        
    def bias(self, X):
        return np.insert(X, 0,1,axis=1)
    
    def log(self, z):
        z=np.array(z,dtype=np.float32)
        return 1.0/(1.0+np.exp(-1.0*z))
    
    def loss (self, y_predict, y_real):
        return (-y_real*np.log1p(y_predict)+(1-y_real)*np.log1p(y_predict)).mean()
        
    def predict_prob(self, X, features):
        
        return np.array(self.log(np.dot(X,features)))
    
    def predict(self, X, feature, threshold =0.5):
        return self.predict_prob(X, feature) >=threshold
        
    def fit(self, X, Y,rate,iter):
        
        feature=[0]*X.shape[1]
        
        for i in range(iter) :
            y_predict = self.predict_prob(X, feature)
            grad = np.dot(X.T, (y_predict - Y))/Y.shape[0]
            feature = feature - rate*grad
        
        return feature

    def evaluate_acc(self, predictY,realY):
        return sum((realY-predictY)**2)
