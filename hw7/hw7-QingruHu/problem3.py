import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

## load data
data = np.load('toy_mnist.npz')

X_train_ = data['X_train'] / 255
X_train = X_train_.reshape(X_train_.shape[0], 784)
X_test_ = data['X_test'] / 255
X_test = X_test_.reshape(X_test_.shape[0], 784)

Y_train_ = data['y_train']
Y_train = np.argmax(Y_train_, axis=1)
Y_test_ = data['y_test']
Y_test = np.argmax(Y_test_, axis=1)


## find the elbow point
Je = []
n = np.arange(1, 55, 5)
for i in n:
    kmeans = KMeans(n_clusters=i).fit(X_train)
    Je.append([kmeans.inertia_])

plt.plot(n, Je)
plt.xlabel('Number of Clusters')
plt.ylabel('Je')
plt.savefig('report/Je.png')
plt.show()
plt.cla()


## Kmeans
nc = 20
# train n_clusters
model = KMeans(init = 'random', n_clusters=nc, n_init=20).fit(X_train)
# find the cluster label
means = model.cluster_centers_
center = []
for i in range(0, nc):
    temp = []
    for j in range(0, 10):
        ind = (Y_train == j)
        p = model.predict(X_train[ind, :])
        t = p == i
        temp.append(np.sum(t))
    temp = np.array(temp)
    center.append(np.argmax(temp))
center = np.array(center)
# plot the cluster centers
x = np.arange(0, 28)
y = np.arange(28, 0, -1) - 1
xs, ys = np.meshgrid(x, y)
for n,i in enumerate(model.cluster_centers_):
    z = i.reshape([28, 28])
    plt.contourf(xs, ys, z)
    cb = plt.colorbar()
    plt.savefig('kmeans/%s.png'%n)
    plt.cla()
    cb.remove()
# predict
Y_pred_ = model.predict(X_test)
Y_pred = np.arange(0, 200)
for i in range(0, 200):
    Y_pred[i] = center[Y_pred_[i]]
ind = (Y_pred == Y_test)
print('Prediction Accuracy of KMeans: ', np.sum(ind) / Y_pred.shape[0])


## EM
model = GaussianMixture(n_components=nc, covariance_type='full', tol=1e-8).fit(X_train)
means = model.means_
center = []
for i in range(0, nc):
    temp = []
    for j in range(0, 10):
        ind = (Y_train == j)
        p = model.predict(X_train[ind, :])
        t = p == i
        temp.append(np.sum(t))
    temp = np.array(temp)
    center.append(np.argmax(temp))
center = np.array(center)
# plot the cluster centers
x = np.arange(0, 28)
y = np.arange(28, 0, -1) - 1
xs, ys = np.meshgrid(x, y)
for n,i in enumerate(model.means_):
    z = i.reshape([28, 28])
    plt.contourf(xs, ys, z)
    cb = plt.colorbar()
    plt.savefig('em/%s.png'%n)
    plt.cla()
    cb.remove()
# predict
Y_pred_ = model.predict(X_test)
Y_pred = np.arange(0, 200)
for i in range(0, 200):
    Y_pred[i] = center[Y_pred_[i]]
ind = Y_pred == Y_test
print('Prediction Accuracy of EM', np.sum(ind) / Y_pred.shape[0])
