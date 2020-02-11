import pandas as pd
import numpy as np
from math import pi
from math import exp
from random import randrange

class NaiveBayes:
    p_firstoutcome = 0
    p_secondoutcome = 0
    
    # two dimensional arrays for mean and standard deviation values
    #firstoutcome_mean = pd.DataFrame()
    #firstoutcome_std = pd.DataFrame()
    #secondoutcome_mean = pd.DataFrame()
    #secondoutcome_std = pd.DataFrame()
    
    firstoutcome_mean = list()
    firstoutcome_std = list()
    secondoutcome_mean = list()
    secondoutcome_std = list() 
    
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
        #firstoutcome, secondoutcome = y.value_counts()
        
        ## Calculate prior probability
        
        # Proportions of each target
        firstoutcome = np.sum(y==0)
        secondoutcome = np.sum(y==1)
        total = firstoutcome + secondoutcome
        
        # Record prior probability
        self.p_firstoutcome = firstoutcome/total
        self.p_secondoutcome = secondoutcome/total
        
        ## Calculate posterior probability
        pos = np.argwhere(y == 1)
        neg = np.argwhere(y == 0)
        
        # Mean and standard deviation for each class-feature pair
        for i in range(X.shape[1]):
            self.firstoutcome_mean.append(X[neg,i].mean())
            self.firstoutcome_std.append(X[neg,i].std())
            self.secondoutcome_mean.append(X[pos,i].mean())
            self.secondoutcome_std.append(X[pos,i].std())
        
        
        
        #df = pd.concat([X, y], axis=1)
        #df_new= pd.concat([X, y], axis=1)
        
        ## where 0 for bad, 1 for good
        #self.firstoutcome = df[Result == '0']
        #self.secondoutcome = df[Result == '1']
        
        ## not sure about these 2 lines????
        #self.firstoutcome = self.firstoutcome.iloc[0:train_num,0:-1]
        #self.secondoutcome = self.secondoutcome.iloc[0:train_num,0:-1]
        
        #self.firstoutcome_mean = self.firstoutcome.mean()
        #self.firstoutcome = self.firstoutcome.std()
        #self.secondoutcome_mean = self.secondoutcome.mean()
        #self.secondoutcome_std = self.secondoutcome.std()
        
        return
    
    # ymean for mean of variable and yvariance for variance of variable
    def Gaussian(self, x, ymean, yvariance):
        p = 1/(np.sqrt(2*np.pi*yvariance)) * np.exp((-(x-ymean)**2)/(2*yvariance)) 
        return p
    
    def predict(self, X):
        ypredict = []
        # get the dimension of input points
        row, column = X.shape  
      
        for i in range(row):
            finalp0 = np.log(self.p_firstoutcome)
            finalp1 = np.log(self.p_secondoutcome)
            for j in range(column):
                #p0 = self.Gaussian(X.iloc[i,j], self.firstoutcome_mean.iloc[j], self.firstoutcome_std.iloc[j])
                #p1 = self.Gaussian(X.iloc[i,j], self.secondoutcome_mean.iloc[j], self.secondoutcome_std.iloc[j])
                if self.firstoutcome_std[j] == 0:
                    p0 = 1
                else:
                    p0 = self.Gaussian(X[i,j], self.firstoutcome_mean[j], (self.firstoutcome_std[j])**2)
                
                if self.secondoutcome_std[j] == 0:
                    p1 = 1
                else:
                    p1 = self.Gaussian(X[i,j], self.secondoutcome_mean[j], (self.secondoutcome_std[j])**2)     
                
                if p0 == 0:
                    p0 = 10**-20
                if p1 == 0:
                    p1 = 10**-20
                finalp0 += np.log(p0)
                finalp1 += np.log(p1)
            
            if finalp0>finalp1:
                ypredict.append(0)
            else:
                ypredict.append(1)
                
        ypredict = np.array(ypredict)
        return ypredict 
    
    def evaluate_acc(self, actual, ypredict):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == ypredict[i]:
                correct += 1
        return correct / float(len(actual))
    
    ## Split dataset into k folds
    #def crossvalidation(dataset, kfolds):
        #split = list()
        #copy = list(dataset)
        #ksize = int(len(dataset) / kfolds)
        #for _ in range(kfolds):
            #fold = list()
            #while len(fold) < ksize:
                #index = randrange(len(copy))
                #fold.append(copy.pop(index))
            #split.append(fold)
        #return split
    
    ## Evaluate using cross validation split
    #def evaluate_algorithm(dataset, algorithm, kfolds, *args):
        #folds = crossvalidation(dataset, kfolds)
        #scores = list()
        #for fold in folds:
            #train_set = list(folds)
            #train_set.remove(fold)
            #train_set = sum(train_set, [])
            #test_set = list()
            #for row in fold:
                #row_copy = list(row)
                #test_set.append(row_copy)
                #row_copy[-1] = None
            #predicted = algorithm(train_set, test_set, *args)
            #actual = [row[-1] for row in fold]
            #accuracy = accuracy_metric(actual, predicted)
            #scores.append(accuracy)
        #return scores
        
    # K-fold partitioning    
    def partition (self, X, Y, K=3):
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
    
    def cross_validation(self, X, Y, K=5):
        split_X, split_Y = self.partition(X, Y, K)
        #log = log_regression(0.1,200)
        score = [0]*(K)
        
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
            feature = self.fit(trainingX, trainingY)
            
            # Test data on the testing set
            fit_y = self.predict(split_X[i])
            score[i] = self.evaluate_acc(fit_y, split_Y[i])
        return np.mean(score)        