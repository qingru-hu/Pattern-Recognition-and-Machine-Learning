import numpy as np
from matplotlib import pyplot as plt

## define some functions
def func(x):
    return 1 / (1+np.exp(-x))

def loss_func(w, b, X, y):
    x = np.dot(X, w) + b
    L = 1/2 * np.sum((func(x) - y)**2)
    return L

def update(X, y, wt, bt, rho=1e-3):
    x = np.dot(X, wt) + bt
    pw = np.dot((func(x) - y) * func(x) * (1-func(x)), X)
    pb = np.sum((func(x) - y) * func(x) * (1-func(x)))
    w = wt - rho*pw
    b = bt - rho*pb
    return w, b

def initial():
    w0 = np.random.normal(0, 1, 9)
    return w0, 0

## set the regression parms
if_ite_else_upl = True
iteration = 10000
uplimit = 1
threshold = 0.5

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

## split the features and labels
X = data[:,1:10]
y = data[:, -1]
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
print('')

## start regression
np.random.seed(10)
w, b = initial()
L = loss_func(w, b, X, y)
Ls = []
if if_ite_else_upl:
    for i in range(iteration):
        w, b = update(X, y, w, b)
        L = loss_func(w, b, X, y)
        print(L)
        Ls.append(L)
else:
    while(L>uplimit):
        w, b = update(X, y, w, b)
        L = loss_func(w, b, X, y)
        print(L)
        Ls.append(L)

## calculate the classification accuracy
print('The best weight: ', np.round(w, 3))
print('The best bias: ', np.round(b, 3))
y_pre = func(np.dot(X, w) + b)
ind_TP = (y==0)*(y_pre<threshold)
ind_TN = (y==1)*(y_pre>threshold)
acc = (np.sum(ind_TP) + np.sum(ind_TN)) / len(y)
print('The classsification accuracy is {:.2f} %.'.format(acc*100))

## plot loss func over regression
plt.plot(range(len(Ls)), Ls)
plt.ylabel('Loss Function')
plt.xlabel('Steps')
if if_ite_else_upl:
    plt.title(f'Max Iterations: {iteration}')
else:
    plt.title(f'Loss Function Upperlimt:{uplimit}')
plt.savefig('regression_loss_func.png', dpi=100)
plt.show()
