# -*- coding: utf-8 -*-
from log_regression2 import log_regression

import numpy as np
from random import randrange
class cross_validation:
    def __init__(self, rate, k = 5, threshold = False, max_iterations = 1000):
        self.K = k
        self.rate = rate
        self.threshold = threshold
        self.max_iterations = 1000
    
    # K-fold partitioning    
    def partition (self, X, Y):
        K = self.K
        fold_size = int(X.shape[0]/K)
        X_split = list()
        copyX = list(X)
        Y_split = list()
        copyY = list(Y)
        
        # Make 3 folds of size fold_size
        for i in range(K):
            foldX = list()
            foldY = list()
            while len(foldX) < fold_size:
                # Choose random elements to be part of the folds
                index = randrange(len(copyX))
                foldX.append(copyX.pop(index))
                foldY.append(copyY.pop(index))
            X_split.append(foldX)
            Y_split.append(foldY)
        return (np.array(X_split),np.array(Y_split))
    
    def evaluate_log(self, X, Y):
        K = self.K
        threshold = self.threshold
        split_X, split_Y = self.partition(X, Y)
        log = log_regression(self.rate,self.max_iterations)
        score = [0]*(K)
        iterations = [0]*(K)
        
        # i = the testing set
        for i in range(K):
            trainingX = []
            trainingY = []
            
            # Make training set from all partitions beside the testing one
            for j in range(K):
                if i != j:
                    trainingX.extend(split_X[j])
                    trainingY.extend(split_Y[j])
            
            #setX = split_X[i]
            #setY = split_Y[i]
            
            trainingX = np.array(trainingX)
            trainingY = np.array(trainingY)
            
            
            # Train data
            
            if threshold == False:
                feature = log.fit(trainingX, trainingY)
            else:
                feature, iterations[i] = log.fit_threshold(trainingX, trainingY)
            
            # Test data on the testing set
            fit_y = log.predict(split_X[i], feature)
            score[i] = log.evaluate_acc(fit_y, split_Y[i])
        
        if threshold == False:
            return np.mean(score)
        else:
            return np.mean(score), np.mean(iterations)