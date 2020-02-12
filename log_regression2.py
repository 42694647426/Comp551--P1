# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from sklearn.model_selection import train_test_split

class log_regression:
    
    def __init__(self, rate, gradient_iter):
        self.rate = rate
        self.gradient_iter = gradient_iter
    
    # Adds bias term    
    def bias(self, X):
        return np.insert(X, 0,1,axis=1)
    
    # Logistic function
    def log(self, z):
        z=np.array(z,dtype=np.float32)
        return 1.0/(1.0+np.exp(-1.0*z))
    
    # Loss Function
    def loss (self, y_predict, y_real):
        return (-y_real*np.log1p(y_predict)+(1-y_real)*np.log1p(y_predict)).mean()
    
    # Generate prediction value between 0 and 1    
    def predict_prob(self, X, features):
        
        return np.array(self.log(np.dot(X,features)))
    
    # predict the binary category
    def predict(self, X, feature, threshold =0.5):
        return self.predict_prob(X, feature) >=threshold
    
    # Train dataset    
    def fit(self, X, Y):
        rate = self.rate
        iterations = self.gradient_iter
        
        feature=[0]*X.shape[1]
        
        for i in range(iterations) :
            y_predict = self.predict_prob(X, feature)
            grad = np.dot(X.T, (y_predict - Y))/Y.shape[0]
            #grad = np.dot(X.T, (y_predict - Y))
            if i != 0:
                change_loss = self.loss(old_y_predict, Y) - self.loss(y_predict, Y)
                #print(change_loss)
            #print(sum(grad**2))
            
            feature = feature - rate*grad
            old_y_predict = y_predict
        
        return feature
    
    # Train dataset with threshold   
    def fit_threshold(self, X, Y, threshold = 5*10**-5):
        rate = self.rate
        iter = self.gradient_iter
        
        feature=[0]*X.shape[1]
        
        max_iterations = iter
        
        # In case of low convergence or divergence, have a limited number of iterations
        for i in range(iter) :
            y_predict = self.predict_prob(X, feature)
            grad = np.dot(X.T, (y_predict - Y))/Y.shape[0]
            #grad = np.dot(X.T, (y_predict - Y))
            if i != 0:
                change_loss = abs(self.loss(old_y_predict, Y) - self.loss(y_predict, Y))
                
                if change_loss < threshold:
                    max_iterations = i+1
                    break
                #print(change_loss)
            #print(sum(grad**2))
            
            feature = feature - rate*grad
            old_y_predict = y_predict.copy()
        
        return feature, max_iterations
    
    # Evaluate the prediction. Gives the proportion of accurate results
    def evaluate_acc(self, realY, predictY):
        return 1 - (sum((realY-predictY)**2)/len(realY))