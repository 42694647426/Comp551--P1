import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv("iono.csv", header=None)

# Transform data and targets in numpy arrays
data = data.as_matrix()
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
plt.hist(res) 
plt.title("Distribution of the positive vs negative classes") 
plt.show()

# X = data
# Y = res