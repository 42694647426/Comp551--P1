import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from log_regression import log_regression
from NaiveBayes import NaiveBayes
from cross_validation import cross_validation
# ionosphere data

data = pd.read_csv("iono.csv", header=None)

# Transform data and targets in numpy arrays
#data = data.as_matrix()
data = data.values
res = np.zeros(len(data))

# Transform targets in a binary representation
for i in range(len(data)):
    if data[i][len(data[0])-1] == 'g':
        res[i]=1
    elif data[i][len(data[0])-1] == 'b':
        res[i]=0

# delete target from data (last column from data)
data = np.delete(data, len(data[0])-1, 1)

#print(data)

# Detect oddities
for i in range(len(data)):
    # Test for missing data
    if len(data[0]) != len(data[i]):
        print("values on row ",i, " are missing")
    
    # Test for malformed features (non-float or integer)
    for j in range(len(data[0])):
        if not(isinstance(data[i][j], int) or isinstance(data[i][j], float)):
            print("value at coordinates",(i,j), "is malformed")

# Histogram of the targets
plt.figure(1)
#plt.hist(res) 
plt.hist([res[np.argwhere(res == 0)], res[np.argwhere(res == 1)]], label=['bad', 'good'])
plt.legend(loc='upper right') 
plt.title("Distribution of the positive vs negative classes") 
plt.show()

# Distributions of some numerical features (feature columns 2,3,4,5 were considered)
pos = np.argwhere(res == 1)
neg = np.argwhere(res == 0)

# matrices (feature, data point) - separation between positive and negative features
pos_features = np.zeros((4,len(pos))) 
neg_features = np.zeros((4,len(neg)))
for i in range(4):
    neg_features[i,:] = np.squeeze(data[neg, i+2])
    pos_features[i,:] = np.squeeze(data[pos, i+2])


plt.figure(2)

for i in range(4):
    plt.subplot(2,2,i+1)
    
    # Set bin boundaries by the minimum and maximum values of the features
    bins = np.linspace(min(min(neg_features[i,:]), min(pos_features[i,:])),
                       max(max(neg_features[i,:]), max(pos_features[i,:])), 30)
    
    # Plot the histogram of the positive and negative features
    plt.hist([neg_features[i,:], pos_features[i,:]], bins, label=['neg', 'pos'])
    plt.legend(loc='upper right')    
    
    plt.title("Distribution of feature #" + str(i+2))

plt.show()


# Correlation between some numerical features (feature columns 2,3,4,5 were considered)

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
    
    plt.title("Correlation between feature #" + str(i+2) + " and #" + str(((i+1)%4)+2))    
    
plt.show()

# Final data variables X and target variables Y
X = np.array(data)
Y = np.array(res)


#fit model
log_model  = log_regression(0.01, 150)
X = log_model.bias(X)
fit_iono  = log_model.fit(X,Y)
pre = log_model.predict(X,fit_iono)
loss = log_model.loss(pre,Y)
print(fit_iono, loss)
validation  = cross_validation(3)
score = validation.evaluate_log(X,Y)
print(score)

