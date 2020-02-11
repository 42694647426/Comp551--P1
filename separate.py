import numpy as np
import random
from random import randrange

# Separate the train and test sets    
def separate (X, Y, split=0.1, seed = 0):
    random.seed(0)
    fold_size = int(X.shape[0]*split)
    X_test = list()
    X_train = list(X)
    Y_test = list()
    Y_train = list(Y)
    
    # Make 3 folds of size fold_size
    #for i in range(K):
    #    foldX = list()
    #    foldY = list()
    while len(X_test) < fold_size:
        # Choose random elements to be part of the folds
        index = randrange(len(X_train))
        X_test.append(X_train.pop(index))
        Y_test.append(Y_train.pop(index))
    
    return (np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test))