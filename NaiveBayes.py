#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

class NaiveBayes:
    p_firstoutcome = 0
    p_secondoutcome = 0
    
    #two dimensional arrays for mean and standard deviation values
    firstoutcome_mean = pd.DataFrame()
    firstoutcome_std = pd.DataFrame()
    secondoutcome_mean = pd.DataFrame()
    secondoutcome_std = pd.DataFrame()
    
    def fit(self,X,y):
        #count the amount of same values 
        firstoutcome, secondoutcome = y.value_counts()
        total = firstoutcome + secondoutcome
        
        self.p_firstoutcome = firstoutcome/total
        self.p_secondoutcome = secondoutcome/total
        
        df = pd.concat([X, y], axis=1)
        
        #where 0 for bad, 1 for good
        self.firstoutcome = df[Result == '0']
        self.secondoutcome = df[Result == '1']
        
        #not sure about these 2 lines????
        self.firstoutcome = self.firstoutcome.iloc[0:train_num,0:-1]
        self.secondoutcome = self.secondoutcome.iloc[0:train_num,0:-1]
        
        self.firstoutcome_mean = self.firstoutcome.mean()
        self.firstoutcome = self.firstoutcome.std()
        self.secondoutcome_mean = self.secondoutcome.mean()
        self.secondoutcome_std = self.secondoutcome.std()
        
        return self
    
    #ymean for mean of variable and yvariance for variance of variable
    def Gaussian(self, x, ymean, yvariance):
        p = 1/(np.sqrt(2*np.pi*yvariance)) * np.exp((-(x-ymean)**2)/(2*yvariance)) 
        return p
    
    def predict(self, X):
        ypredict = []
        #get the dimension of input points
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


# In[ ]:




