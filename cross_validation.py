# -*- coding: utf-8 -*-
from log_regression import log_regression

import numpy as np
from random import randrange
class cross_validation:
    def __init__(self, k):
        self.K = k
        
    def partition (self, X, K=3):
         fold_size = int(X.shape[0]/K)
         X_split = list()
         copy = list(X)
         for i in range(K):
             fold = list()
             while len(fold) < fold_size:
                 index = randrange(len(copy))
                 fold.append(copy.pop(index))
             X_split.append(fold)
         return np.array(X_split)
    
    def evaluate_log(self, X, Y, K=3):
        split_X = self.partition(X, K)
        split_Y = self.partition(Y, K)
        log = log_regression(0.1,200)
        score = [0]*(K-1)
        for i in range(K-1):
            set = split_X[i]
            setY = split_Y[i]
            feature = log.fit(set, setY)
            fit_y = log.predict(split_X[K-1], feature)
            score[i] = log.loss(fit_y, split_Y[K-1])
        return np.mean(score)