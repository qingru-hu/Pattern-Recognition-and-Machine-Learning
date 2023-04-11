import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '8'
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def class1():
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    return np.random.multivariate_normal(mean, cov)

def class2():
    mean = np.array([2, 2])
    cov = np.array([[1, 0], [0, 1]])
    return np.random.multivariate_normal(mean, cov)

def generate_data(n):
    X = []
    for i in range(0, n):
        t = np.random.rand()
        if t <= 2 / 3:
            X.append(class1())
        else:
            X.append(class2())
    X = np.array(X)
    return X

# set n = 2000, change t
# set tol = 1e-114514 to make the interation times reach the set interation times.
n = 2000
data = generate_data(n)

# find the relation between iteration times and error
t_err_00 = []
t_err_22 = []
# times = np.linspace(100, 500, 50, dtype=int)
times = np.array([5, 10, 20, 50, 100, 150, 200, 300])

for t in times:  
    model = GaussianMixture(n_components=2, covariance_type='full', tol=1e-114514, max_iter=t)
    model.fit(data)
    if model.means_[0][0] + model.means_[0][1] > model.means_[1][0] + model.means_[1][1]:
        t_err_22.append(((model.means_[0][0] - 2)**2 + (model.means_[0][1] - 2)**2)**0.5)
        t_err_00.append(((model.means_[1][0] - 0)**2 + (model.means_[1][1] - 0)**2)**0.5)
    else:
        t_err_00.append(((model.means_[0][0] - 2)**2 + (model.means_[0][1] - 2)**2)**0.5)
        t_err_22.append(((model.means_[1][0] - 0)**2 + (model.means_[1][1] - 0)**2)**0.5)

plt.scatter(times, t_err_00, label='center at (0, 0)')
plt.scatter(times, t_err_22, label='center at (2, 2)')
plt.xlabel('Interation Times t')
plt.ylabel('Error')
plt.legend()
plt.savefig('report/iterate_t.png', dpi=200)
plt.cla()

# find the relation betweennumber of data points and error
# n_err_00 = []
# n_err_22 = []
# numbers = np.linspace(100, 10000, 10, dtype=int)

# for n in numbers: 
#     data = generate_data(n) 
#     model = GaussianMixture(n_components=2, covariance_type='full', tol=1e-114514, max_iter=500)
#     model.fit(data)
#     if model.means_[0][0] + model.means_[0][1] > model.means_[1][0] + model.means_[1][1]:
#         n_err_22.append(((model.means_[0][0] - 2)**2 + (model.means_[0][1] - 2)**2)**0.5)
#         n_err_00.append(((model.means_[1][0] - 0)**2 + (model.means_[1][1] - 0)**2)**0.5)
#     else:
#         n_err_00.append(((model.means_[0][0] - 0)**2 + (model.means_[0][1] - 0)**2)**0.5)
#         n_err_22.append(((model.means_[1][0] - 2)**2 + (model.means_[1][1] - 2)**2)**0.5)

# plt.plot(numbers, n_err_00, label='center at (0, 0)')
# plt.plot(numbers, n_err_22, label='center at (2, 2)')
# plt.xlabel('Number of Steps N')
# plt.ylabel('Error')
# plt.legend()
# plt.savefig('report/iterate_n.png', dpi=200)