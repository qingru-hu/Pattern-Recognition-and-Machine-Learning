import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def dist1():
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    return np.random.multivariate_normal(mean, cov)


def dist2():
    mean = np.array([2, 2])
    cov = np.array([[1, 0], [0, 1]])
    return np.random.multivariate_normal(mean, cov)


def dataLoader(n:int):
    X = np.empty([0, 2])
    for i in range(0, n):
        t = np.random.rand()
        if t <= 2 / 3:
            X = np.append(X, [dist1()], axis = 0)
        else:
            X = np.append(X, [dist2()], axis = 0)

    return X

# set n = 2000, change t
# set tol = 1e-114514 to make the interation times reach the max interation times.
data = dataLoader(2000)
et00 = np.array([])
et22 = np.array([])
times = np.array([5, 10, 20, 50, 100, 150, 200, 300])

for t in times:  
    model = GaussianMixture(n_components=2, covariance_type='full', tol=1e-114514, max_iter=t)
    model.fit(data)
    if model.means_[0][0] + model.means_[0][1] > model.means_[1][0] + model.means_[1][1]:
        et22 = np.append(et22, ((model.means_[0][0] - 2)**2 + (model.means_[0][1] - 2)**2)**0.5)
        et00 = np.append(et00, ((model.means_[1][0] - 0)**2 + (model.means_[1][1] - 0)**2)**0.5)
    else:
        et00 = np.append(et00, ((model.means_[0][0] - 0)**2 + (model.means_[0][1] - 0)**2)**0.5)
        et22 = np.append(et22, ((model.means_[1][0] - 2)**2 + (model.means_[1][1] - 2)**2)**0.5)

plt.plot(times, et00, label='(0, 0)')
plt.plot(times, et22, label='(2, 2)')
plt.xlabel('interation times')
plt.ylabel('error')
plt.legend()
plt.savefig('interation_times.png')
plt.cla()

# find the connection between
en00 = np.array([])
en22 = np.array([])
numbers = np.array([100, 300, 500, 1000, 2000, 5000, 10000])

for n in numbers: 
    data = dataLoader(n) 
    model = GaussianMixture(n_components=2, covariance_type='full', tol=1e-114514, max_iter=500)
    model.fit(data)
    if model.means_[0][0] + model.means_[0][1] > model.means_[1][0] + model.means_[1][1]:
        en22 = np.append(en22, ((model.means_[0][0] - 2)**2 + (model.means_[0][1] - 2)**2)**0.5)
        en00 = np.append(en00, ((model.means_[1][0] - 0)**2 + (model.means_[1][1] - 0)**2)**0.5)
    else:
        en00 = np.append(en00, ((model.means_[0][0] - 0)**2 + (model.means_[0][1] - 0)**2)**0.5)
        en22 = np.append(en22, ((model.means_[1][0] - 2)**2 + (model.means_[1][1] - 2)**2)**0.5)

plt.plot(numbers, en00, label='(0, 0)')
plt.plot(numbers, en22, label='(2, 2)')
plt.xlabel('number of steps')
plt.ylabel('error')
plt.legend()
plt.savefig('number_of_samples.png')

data = dataLoader(5000) 
model = GaussianMixture(n_components=2, covariance_type='full', tol=1e-8, max_iter=100)
model.fit(data)

print('means: ', model.means_)
print('covariances: ', model.covariances_)
print('weights: ', model.weights_)