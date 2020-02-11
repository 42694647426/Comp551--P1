
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from log_regression2 import log_regression
from NaiveBayes3 import NaiveBayes
from cross_validation2 import cross_validation
import separate
import seaborn as sns
import string

class gradient():
    def grad(X, y, w):
        N,D = X.shape
        model = log_regression(0.2,200)
        yh = model.log(np.dot(X, w)) 
        grad = np.dot(X.T, yh - y) / N 
        return grad
    
    def GradientDescent(X, y, lr, eps):
        N,D = X.shape 
        w = np.zeros(D)     
        g = np.inf 
        while np.linalg.norm(g) > eps: 
            g = gradient.grad(X, y, w) 
            w = w - lr*g
            print(g)
            return w

