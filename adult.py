import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from log_regression2 import log_regression
from NaiveBayes3 import NaiveBayes
from cross_validation2 import cross_validation
import separate

# dictionary of categories: contains a dictionary for each feature, mapping an integer
# to each category
categories = {}

# Makes all the inputs of the dataset numerical
def transform(data):
    for j in range(len(data[0])):
        
        
        category_number = 0 # keeps track of the category number in the feature
        for i in range(len(data)):
            # Ignore features that are already numerical
            if (isinstance(data[i][j], int) or isinstance(data[i][j], float)):
                break
            
            # Create a dictionary for a feature if it does not exist
            if j not in categories:
                categories[j] = {}
            
            # Add a numerical value to each category
            if data[i][j] not in categories[j]:
                categories[j][data[i][j]] = category_number
                category_number += 1
            
            data[i][j] = categories[j][data[i][j]]

# Makes data one-hot encoded
def oneHot(data):      
    one_hot = []  
    for i in range(len(data)):
        one_hot.append(list())
        
        for j in range(len(data[0])):
            # Just copy non-categorical features
            if j not in categories:
                one_hot[i].append(data[i][j])
            # Make categorical features one-hot encoded
            else:
                # Array of 0s of the length of the categories in that feature
                temp = [0]*len(categories[j])
                
                # Put 1 to the correct category
                temp[data[i][j]] = 1
                
                one_hot[i].extend(temp)
                
    one_hot = np.array(one_hot)            
    return one_hot           
                
    


data = pd.read_csv("adult.csv", header=None)

# Transform data in numpy arrays
data = data.values

data_temp = np.zeros(data.shape, dtype = 'O')
        
row = 0
# Detect oddities
for i in range(len(data)):
    
    # Test for missing data (row length)
    if len(data[0]) != len(data[i]):
        print("values on row ",i, " are missing")
    
    # Test for malformed features
    include = True # Only include rows that don't have missing features (?)
    for j in range(len(data[0])):
        # remove leading/trailing spaces
        if isinstance(data[i][j], str):
            data[i][j] = data[i][j].strip()
            
        # remove instances with missing data
        if data[i][j] == "?":
            include = False
            break
        
    # Only include rows that don't have missing features (?)
    if include == True:
        data_temp[row,:] = data[i,:]
        row += 1        

data = data_temp[0:row] 
data_raw = data.copy()

# Transform data into numerical values
transform(data)


res = np.zeros(len(data))

# Transform targets in a binary representation
for i in range(len(data)):
    res[i] = data[i][len(data[0])-1]
    
#    if data[i][len(data[0])-1] == ">50K":
#        res[i]=1
#    elif data[i][len(data[0])-1] == "<=50K":
#        res[i]=0

# delete target from data (last column from data)
data = np.delete(data, len(data[0])-1, 1)

# Make the input categories one_hot encoded
one_hot = oneHot(data)

#print(data)

# Histogram of the targets
plt.figure(1)
#plt.hist(res) 
plt.hist([res[np.argwhere(res == 0)], res[np.argwhere(res == 1)]], label=['<=50K', '>50K'])
plt.legend(loc='upper right') 
plt.title("Distribution of the positive vs negative classes") 
plt.show()

# Distributions of some numerical features (feature columns 0,2,4,10 were considered)
f = (0,2,4,10)

pos = np.argwhere(res == 1)
neg = np.argwhere(res == 0)

# matrices (feature, data point) - separation between positive and negative features
pos_features = np.zeros((4,len(pos))) 
neg_features = np.zeros((4,len(neg)))
for i in range(4):
    neg_features[i,:] = np.squeeze(data[neg, f[i]])
    pos_features[i,:] = np.squeeze(data[pos, f[i]])


plt.figure(2)

for i in range(4):
    plt.subplot(2,2,i+1)
    
    # Set bin boundaries by the minimum and maximum values of the features
    bins = np.linspace(min(min(neg_features[i,:]), min(pos_features[i,:])),
                       max(max(neg_features[i,:]), max(pos_features[i,:])), 30)
    
    # Plot the histogram of the positive and negative features
    plt.hist([neg_features[i,:], pos_features[i,:]], bins, label=['neg', 'pos'])
    plt.legend(loc='upper right')    
    
    plt.title("Distribution of feature #" + str(f[i]))

plt.show()


# Correlation between some numerical features (feature columns 0,2,4,10 were considered)

plt.figure(3)

for i in range(4):
    plt.subplot(2,2,i+1)
    
    # Correlation coefficients
    r_neg = np.corrcoef(neg_features[i,:], neg_features[(i+1)%4,:])
    r_pos = np.corrcoef(pos_features[i,:], pos_features[(i+1)%4,:])
    
    # Labels for the legend
    lbl_neg = "r_neg = " + str(round(r_neg[0,1],4))
    lbl_pos = "r_pos = " + str(round(r_pos[0,1],4))
    
    plt.scatter(neg_features[i,:], neg_features[(i+1)%4,:], label=lbl_neg)
    plt.scatter(pos_features[i,:], pos_features[(i+1)%4,:], label=lbl_pos)
    
    plt.legend(loc='upper right')    
    
    plt.title("Correlation between feature #" + str(f[i]) + " and #" + str(f[(i+1)%4]))   
    

plt.show()

# Final data variables X and target variables Y
X = np.array(one_hot)
Y = np.array(res)

## Compare accuracy of naive Bayes and logistic regression before finding best learning rate

# All datasets will use for logistic regression the same learning rate = 0.01 and # iterations = 500
rate = 0.01
iterations = 500

log_model  = log_regression(rate, iterations)
X = log_model.bias(X) # add bias column

# Separate training and testing sets 
X_train, Y_train, X_test, Y_test = separate.separate(X,Y)


## Logistic regression

# train the data
fit_iono  = log_model.fit(X_train,Y_train) 

# Cross validation
validation  = cross_validation(rate, max_iterations = 10000)
score = validation.evaluate_log(X_train,Y_train)
print("Averaged training accuracy for Logistic Regression: ", score)

# Test data
pre = log_model.predict(X_test,fit_iono) 
acc = log_model.evaluate_acc(pre,Y_test)
print("Accuracy on testing data for Logistic Regression: ", acc)

## Naive Bayes

# train the data



# Cross validation
bayes_model = NaiveBayes()
score = bayes_model.cross_validation(X_train,Y_train)
print("Averaged training accuracy for Naive Bayes: ", score)


# Test data
bayes_model = NaiveBayes()
fit_bayes = bayes_model.fit(X_train,Y_train)
pre = bayes_model.predict(X_test)
acc = bayes_model.evaluate_acc(pre,Y_test)
print("Accuracy on testing data for Naive Bayes: ", acc)

print()
## Test different learning rates for gradient descent

# Loss function threshold = 5*10^-5; maximum number of iterations = 1000

# Find automatically the best rate by choosing the rate that gives the best
# validation accuracy with the lowest number of iterations
best_rate = 0
best_accuracy = 0
lowest_iterations = 0

acc = []
iters = []
rate = 10**-15
for i in range(20):
    
    # Cross validation
    validation  = cross_validation(rate, threshold = True)
    score, iterations = validation.evaluate_log(X_train,Y_train)
    #print("Averaged training accuracy for Logistic Regression: ", score)
    print("rate = ", rate, "; iterations = ", iterations, "; accuracy = ", score)
    iters.append(iterations)
    acc.append(score)
    
    if best_accuracy < score:
        best_accuracy = score
        best_rate = rate
        lowest_iterations = iterations
        
    elif best_accuracy == score and lowest_iterations > iterations:
        best_accuracy = score
        best_rate = rate
        lowest_iterations = iterations     
    
    rate *= 10
    
    
plt.figure(4)
plt.scatter(iters, acc)
plt.xlabel("iterations")
plt.ylabel("accurary")
plt.title("the accuracy on train set as a function of iterations of gradient descent")

plt.show()

print()
print("The best learning rate found is: ", best_rate)
print()


## Compare accuracy of naive Bayes and logistic regression before finding best learning rate

rate = best_rate
iterations = 5000

## Logistic regression

# train the data
log_model  = log_regression(rate, iterations)
fit_iono  = log_model.fit(X_train,Y_train) 

# Cross validation
validation  = cross_validation(rate, max_iterations = 10000)
score = validation.evaluate_log(X_train,Y_train)
print("Averaged training accuracy for Logistic Regression: ", score)

# Test data
pre = log_model.predict(X_test,fit_iono) 
acc = log_model.evaluate_acc(pre,Y_test)
print("Accuracy on testing data for Logistic Regression: ", acc)

## Naive Bayes

# train the data



# Cross validation
bayes_model = NaiveBayes()
score = bayes_model.cross_validation(X_train,Y_train)
print("Averaged training accuracy for Naive Bayes: ", score)


# Test data
bayes_model = NaiveBayes()
fit_bayes = bayes_model.fit(X_train,Y_train)
pre = bayes_model.predict(X_test)
acc = bayes_model.evaluate_acc(pre,Y_test)
print("Accuracy on testing data for Naive Bayes: ", acc)

print()







## Accuracy as a function of the size of dataset

# Logistic regression
acc = []
size = []
accT = []
sizeT = []
split_size = 0.1

for i in range(9):
    X_train_size, Y_train_size, X_discard, Y_discard = separate.separate(X_train,Y_train, split=split_size)
    # Cross validation
    validation  = cross_validation(rate, threshold = True)
    score, iterations = validation.evaluate_log(X_train_size,Y_train_size)
    #print("Averaged training accuracy for Logistic Regression: ", score)
    print("CV: size of X = ", X_train_size.shape[0], "; iterations = ", iterations, "; accuracy = ", score)
    
    # Test data
    log_model = log_regression(rate, 500)
    fit_iono  = log_model.fit(X_train_size,Y_train_size)
    pre = log_model.predict(X_test,fit_iono) 
    accur = log_model.evaluate_acc(pre,Y_test)
    print("Test: size of X = ", X_train_size.shape[0], "; accuracy = ", accur)
    #print("Accuracy on testing data for Logistic Regression: ", acc)    
    
    
    size.append(X_train_size.shape[0])
    acc.append(score)
    
    sizeT.append(X_train_size.shape[0])
    accT.append(accur)    
    
    split_size += 0.1

split_size = 0.91
for i in range(8):
    X_train_size, Y_train_size, X_discard, Y_discard = separate.separate(X_train,Y_train, split=split_size)
    # Cross validation
    validation  = cross_validation(rate, threshold = True)
    score, iterations = validation.evaluate_log(X_train_size,Y_train_size)
    #print("Averaged training accuracy for Logistic Regression: ", score)
    print("CV: size of X = ", X_train_size.shape[0], "; iterations = ", iterations, "; accuracy = ", score)
    
    # Test data
    log_model = log_regression(rate, 500)
    fit_iono  = log_model.fit(X_train_size,Y_train_size)
    pre = log_model.predict(X_test,fit_iono) 
    accur = log_model.evaluate_acc(pre,Y_test)
    print("Test: size of X = ", X_train_size.shape[0], "; accuracy = ", accur)
    #print("Accuracy on testing data for Logistic Regression: ", acc)    
    
    
    size.append(X_train_size.shape[0])
    acc.append(score)
    
    sizeT.append(X_train_size.shape[0])
    accT.append(accur)    
    
    split_size += 0.01

log_sizeT = sizeT
log_accT = accT

plt.figure(5)    
plt.plot(size, acc, label = "CV")
plt.plot(sizeT, accT, label = "Test")
plt.legend()
plt.xlabel("size of X_train")
plt.ylabel("accurary")
plt.title("the accuracy on train set as a function of size of X on logistic model")

plt.show()

print()

#bayes model
acc = []
size = []
accT = []
sizeT = []
split_size = 0.1

for i in range(9):
    X_train_size, Y_train_size, X_discard, Y_discard = separate.separate(X_train,Y_train, split=split_size)
    # Cross validation
    bayes_model = NaiveBayes()
    score = bayes_model.cross_validation(X_train_size,Y_train_size)
    print("CV: size of X = ", X_train.shape[0], "; accuracy = ", score)
    
    # Test data
    bayes_model = NaiveBayes()
    fit_bayes = bayes_model.fit(X_train_size,Y_train_size)
    pre = bayes_model.predict(X_test)
    accur = bayes_model.evaluate_acc(pre,Y_test)
    print("Test: size of X = ", X_train_size.shape[0], "; accuracy = ", accur)
    #print("Accuracy on testing data for Logistic Regression: ", acc)    
    
    
    size.append(X_train_size.shape[0])
    acc.append(score)
    
    sizeT.append(X_train_size.shape[0])
    accT.append(accur)    
    
    split_size += 0.1

split_size = 0.91
for i in range(8):
    X_train_size, Y_train_size, X_discard, Y_discard = separate.separate(X_train,Y_train, split=split_size)
    # Cross validation
    bayes_model = NaiveBayes()
    score = bayes_model.cross_validation(X_train_size,Y_train_size)
    print("CV: size of X = ", X_train.shape[0], "; accuracy = ", score)
    
    # Test data
    bayes_model = NaiveBayes()
    fit_bayes = bayes_model.fit(X_train_size,Y_train_size)
    pre = bayes_model.predict(X_test)
    accur = bayes_model.evaluate_acc(pre,Y_test)
    print("Test: size of X = ", X_train_size.shape[0], "; accuracy = ", accur)
    #print("Accuracy on testing data for Logistic Regression: ", acc)    
    
    
    size.append(X_train_size.shape[0])
    acc.append(score)
    
    sizeT.append(X_train_size.shape[0])
    accT.append(accur)    
    
    split_size += 0.01


#for i in range(9):
    #X_train, Y_train, X_test, Y_test = separate.separate(X,Y, split=split_size)
    ## Cross validation
    #score = bayes_model.cross_validation(X_train,Y_train)
    ##print("Averaged training accuracy for Logistic Regression: ", score)
    #print("size of X = ", X_train.shape[0], "; accuracy = ", score)
    #size.append(X_train.shape[0])
    #acc.append(score)
    #split_size += 0.1

plt.figure(6)    
plt.xlabel("size of X_train")
plt.ylabel("accurary")
plt.title("the accuracy on train set as a function of size od X on Naive Bayes model")
plt.plot(size, acc, label = "CV")
plt.plot(sizeT, accT, label = "Test")
plt.legend()
plt.show()

plt.figure(7)    
plt.xlabel("size of X_train")
plt.ylabel("accurary")
plt.title("the accuracy on train set as a function of size od X on Naive Bayes model")
plt.plot(log_sizeT, log_accT, label = "Logistic Regression Test")
plt.plot(sizeT, accT, label = "Naive Bayes Test")
plt.legend()
plt.show()