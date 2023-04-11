import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


data = np.load('toy_mnist.npz')

def dataFlattener(data):
    data_f = np.empty([0, 28*28])
    for i in data:
        data_f = np.append(data_f, [i.flatten()], axis=0)

    return data_f

def labelHandler(label):
    label_h = np.array([])
    for i in label:
        label_h = np.append(label_h, [np.argmax(i)])
    
    return label_h

X_train_ = data['X_train'] / 255
X_test_ = data['X_test'] / 255
Y_train_ = data['y_train']
Y_test_ = data['y_test']

X_train = dataFlattener(X_train_)
Y_train = labelHandler(Y_train_)

X_test = dataFlattener(X_test_)
Y_test = labelHandler(Y_test_)


# find the elbow point
Je = np.array([])
n = np.array([1, 3, 5, 10, 15, 20, 30, 50])
for i in n:
    kmeans = KMeans(n_clusters=i).fit(X_train)
    Je = np.append(Je, [kmeans.inertia_])

plt.plot(n, Je)
plt.xlabel('number of clusters')
plt.ylabel('Je')
plt.savefig('Je.png')
plt.cla()


nc = 20

# train with n_clusters
train_model = KMeans(init = 'random', n_clusters=nc, n_init=20).fit(X_train)

# find the number that each cluster corresponding
means = train_model.cluster_centers_
center = np.empty(nc)

for i in range(0, nc):
    temp = np.empty(10)
    for j in range(0, 10):
        index = Y_train == j
        p = train_model.predict(X_train[index, :])
        t = p == i
        temp[j] = np.sum(t)
    center[i] = np.argmax(temp)


# plot the cluster centers
x = np.arange(0, 28)
y = np.arange(28, 0, -1) - 1
xgrid, ygrid = np.meshgrid(x, y)
n = 0

for i in train_model.cluster_centers_:
    z = i.reshape([28, 28])
    plt.contourf(xgrid, ygrid, z)
    cb = plt.colorbar()
    plt.savefig('Center_of_KMeans/' + str(n) + '.png')
    plt.cla()
    cb.remove()
    n= n + 1


# predict

Y_pred_ = train_model.predict(X_test)
Y_pred = np.arange(0, 200)

for i in range(0, 200):
    Y_pred[i] = center[Y_pred_[i]]

index = Y_pred == Y_test

print('accuracy of KMeans: ', np.sum(index) / Y_pred.shape[0])


# EM

EM = GaussianMixture(n_components=nc, covariance_type='full', tol=1e-6).fit(X_train)

meansEM = EM.means_
centerEM = np.empty(nc)

for i in range(0, nc):
    temp = np.empty(10)
    for j in range(0, 10):
        index = Y_train == j
        p = EM.predict(X_train[index, :])
        t = p == i
        temp[j] = np.sum(t)
    centerEM[i] = np.argmax(temp)


# plot the cluster centers
x = np.arange(0, 28)
y = np.arange(28, 0, -1) - 1
xgrid, ygrid = np.meshgrid(x, y)
n = 0

for i in EM.means_:
    z = i.reshape([28, 28])
    plt.contourf(xgrid, ygrid, z)
    cb = plt.colorbar()
    plt.savefig('Center_of_Em/' + str(n) + '.png')
    plt.cla()
    cb.remove()
    n= n + 1


# predict

Y_pred_ = EM.predict(X_test)
Y_pred = np.arange(0, 200)

for i in range(0, 200):
    Y_pred[i] = centerEM[Y_pred_[i]]

index = Y_pred == Y_test

print('accuracy of EM', np.sum(index) / Y_pred.shape[0])

