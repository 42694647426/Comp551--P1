import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
    
    plt.title("Correlation between feature #" + str(f[i]) + " and #" + str(f[(i+1)%4]))   
    

# Final data variables X and target variables Y
X = data
Y = res

