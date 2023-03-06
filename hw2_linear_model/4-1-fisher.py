import numpy as np
from matplotlib import pyplot as plt


## load data
data = np.genfromtxt('breast-cancer-wisconsin.txt', dtype=int)
data  = data.astype(float)
# the np.genfromtxt load '?' as -1
print('Missing values\' index:')
print(np.array(np.where(data==-1)).T)
# all the missing values are in the 7th column
# fill them with the mean of the 7th column of the same class
good_line = (data[:, 6]==-1)*(data[:, -1]==1)
data[good_line, 6] = np.mean(data[data[:, -1]==1, 6])
bad_line = (data[:, 6]==-1)*(data[:, -1]==0)
data[bad_line, 6] = np.mean(data[data[:, -1]==0, 6])

# split the features and labels
X = data[:,1:10]
y = data[:, -1]
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
print('')

# split the benign and malignant
ind_good = y==0
ind_bad = y==1
X_good = X[ind_good, :]
X_bad = X[ind_bad, :]

print('The mean vector of the benign')
m1 = np.mean(X_good, axis=0)
print(m1)
print('The mean vector of the malignant')
m2 = np.mean(X_bad, axis=0)
print(m2)
print('')

SW = np.dot((X_good - m1).T, (X_good - m1)) + np.dot((X_bad - m2).T, (X_bad - m2))
SW_inv = np.linalg.inv(SW)
w = np.dot(SW_inv, (m2 - m1))
w = w/np.sqrt(np.sum(w**2))
print('The best projection unit vector')
print(np.round(w, 3))
print('')

w0 = -(np.dot(w, m1) + np.dot(w, m2))/2
y_pre = np.dot(X, w)+w0
plt.hist(y_pre[ind_good], bins=30, alpha=0.5, density=True, label='Benign')
plt.hist(y_pre[ind_bad], bins=30, alpha=0.5, density=True, label='Malignant')
plt.legend()
plt.ylabel('Counts')
plt.xlabel('Position on the Projection Plane')
plt.tight_layout()
plt.savefig('fisher_plane.png', dpi=100)
plt.show()
y_pre[y_pre<0] = 0
y_pre[y_pre>0] = 1

ind_TP = (y==1)*(y_pre==1)
ind_TN = (y==0)*(y_pre==0)
acc = (np.sum(ind_TP) + np.sum(ind_TN)) / len(y)
print('The classsification accuracy is {:.5f}.'.format(acc))
