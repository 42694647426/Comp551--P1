#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import pi
from math import exp

class NaiveBayes:
    p_firstoutcome = 0
    p_secondoutcome = 0
    
    # two dimensional arrays for mean and standard deviation values
    firstoutcome_mean = pd.DataFrame()
    firstoutcome_std = pd.DataFrame()
    secondoutcome_mean = pd.DataFrame()
    secondoutcome_std = pd.DataFrame()
    
    # Calculate the mean of a list of numbers
    def mean(numbers):
        return sum(numbers)/float(len(numbers))
 
    # Calculate the standard deviation of a list of numbers
    def std(numbers):
        avg = mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)
    
    
    def fit(self,X,y):
        # count the amount of same values 
        firstoutcome, secondoutcome = y.value_counts()
        total = firstoutcome + secondoutcome
        
        self.p_firstoutcome = firstoutcome/total
        self.p_secondoutcome = secondoutcome/total
        
        df = pd.concat([X, y], axis=1)
        
        # where 0 for bad, 1 for good
        self.firstoutcome = df[Result == '0']
        self.secondoutcome = df[Result == '1']
        
        # not sure about these 2 lines????
        self.firstoutcome = self.firstoutcome.iloc[0:train_num,0:-1]
        self.secondoutcome = self.secondoutcome.iloc[0:train_num,0:-1]
        
        self.firstoutcome_mean = self.firstoutcome.mean()
        self.firstoutcome = self.firstoutcome.std()
        self.secondoutcome_mean = self.secondoutcome.mean()
        self.secondoutcome_std = self.secondoutcome.std()
        
        return self
    
    # ymean for mean of variable and yvariance for variance of variable
    def Gaussian(self, x, ymean, yvariance):
        p = 1/(np.sqrt(2*np.pi*yvariance)) * np.exp((-(x-ymean)**2)/(2*yvariance)) 
        return p
    
    def predict(self, X):
        ypredict = []
        # get the dimension of input points
        row, column = X.shape  
      
        for i in range(row-1):
            finalp0 = np.log(self.p_firstoutcome)
            finalp1 = np.log(self.p_secondoutcome)
            for j in range(column-1):
                p1 = self.Gaussian(X.iloc[i,j], self.firstoutcome_mean.iloc[j], self.firstoutcome_std.iloc[j])
                p2 = self.Gaussian(X.iloc[i,j], self.secondoutcome_mean.iloc[j], self.secondoutcome_std.iloc[j])
                finalp0 = finalp0 + np.log(p1)
                finalp1 = finalp1 + np.log(p2)
            
            if finalp0>finalp1:
                ypredict.append("0")
            else:
                ypredict.append("1")  
        return ypredict 
    
    def accuracy(self, actual, ypredict):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == ypredict[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
    
    # Split dataset into k folds
    def crossvalidation_split(dataset, kfolds):
        split = list()
        copy = list(dataset)
        ksize = int(len(dataset) / kfolds)
        for _ in range(kfolds):
            fold = list()
            while len(fold) < ksize:
                index = randrange(len(copy))
                fold.append(copy.pop(index))
            split.append(fold)
        return split
    
    


# In[ ]:




