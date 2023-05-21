import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cp

def plotelli(mean:np.ndarray, cov:np.ndarray, k, pos, color):
    theta = np.linspace(0, 2*np.pi, 1000)
    Q = np.linalg.cholesky(cov)

    ellipsis = mean + k * Q @ np.array([np.cos(theta), np.sin(theta)])

    plt.plot(ellipsis[0,:], ellipsis[1,:], color=color)
    plt.annotate('k = {x}'.format(x = k), xy=(ellipsis[0,pos], ellipsis[1,pos]), xytext=(ellipsis[0,pos], ellipsis[1,pos]))

def MPMSolver(mean1, mean2, cov1, cov2):
    Q1 = np.linalg.cholesky(cov1)
    Q2 = np.linalg.cholesky(cov2)

    w = cp.Variable(2)
    F0 = (mean1 - mean2)
    F = F0.T

    prob = cp.Problem(cp.Minimize(cp.norm(Q1.T@w) + cp.norm(Q2.T@w)),
                  [F @ w == 1])
    prob.solve()
    w_ = w.value
    k_1 = 1 / (np.sqrt(w_.T @ cov1 @ w_) + np.sqrt(w_.T @ cov2 @ w_))
    return k_1

# define the parameters
mean1 = np.array([[0, 0]]).T
mean2 = np.array([[4, 2]]).T
cov1 = np.array([[4, -1], [-1, 1]])
cov2 = np.array([[1, 1], [1, 2]])

# find the k*
k = MPMSolver(mean1, mean2, cov1, cov2)
print('Best k: ', k)

# plot the ellipsoids
ks = np.arange(0.5, 3, 0.5)
for x in ks:
    plotelli(mean1, cov1, x, 400, 'green')
    plotelli(mean2, cov2, x, 500, 'orange')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('ks.png')
plt.show()
