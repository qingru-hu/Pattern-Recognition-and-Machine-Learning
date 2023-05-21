import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cp

def plotEllli(mean:np.ndarray, cov:np.ndarray, k, pos, color):
    theta = np.linspace(0, 2*np.pi, 1000)
    Q = np.linalg.cholesky(cov)

    ellipsis = mean + k * Q @ np.array([np.cos(theta), np.sin(theta)])

    plt.plot(ellipsis[0,:], ellipsis[1,:], color=color)
    plt.annotate('k = {x}'.format(x = k), xy=(ellipsis[0,pos], ellipsis[1,pos]), xytext=(ellipsis[0,pos], ellipsis[1,pos]))

def geoSolver(mean1, mean2, cov1, cov2):
    theta = np.linspace(0, 2*np.pi, 100000)
    Q1 = np.linalg.cholesky(cov1)
    Q2 = np.linalg.cholesky(cov2)
    e = 1e-3

    for i in tqdm(range(1512000, 1512400)):
        k = 0.000001 * i
        ellipsis1 = mean1 + k * Q1 @ np.array([np.cos(theta), np.sin(theta)])
        ellipsis2 = mean2 + k * Q2 @ np.array([np.cos(theta), np.sin(theta)])

        for p in range(2 * 4140, 2 * 4160):
            for q in range(2 * 28390, 2 * 28410):
                if np.linalg.norm(ellipsis1[:, p] - ellipsis2[:, q]) < e:
                    print(p, q)
                    return k

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
  
mean1 = np.array([[0, 0]]).T
mean2 = np.array([[4, 2]]).T

cov1 = np.array([[4, -1], [-1, 1]])
cov2 = np.array([[1, 1], [1, 2]])

# find the k*
k = geoSolver(mean1, mean2, cov1, cov2)
print('geo: ', k)

k = MPMSolver(mean1, mean2, cov1, cov2)
print('MPM: ', k)

plt.xlabel('x')
plt.ylabel('y')

# plot the elli
plotEllli(mean1, cov1, 0.5, 400, 'r')
plotEllli(mean1, cov1, 1, 400, 'r')
plotEllli(mean1, cov1, 1.512, 400, 'r')
plotEllli(mean1, cov1, 2, 400, 'r')
plotEllli(mean1, cov1, 2.5, 400, 'r')

plotEllli(mean2, cov2, 0.5, 300, 'b')
plotEllli(mean2, cov2, 1, 300, 'b')
plotEllli(mean2, cov2, 1.512, 300, 'b')
plotEllli(mean2, cov2, 2, 300, 'b')
plotEllli(mean2, cov2, 2.5, 300, 'b')

plt.savefig('ellipsoids.png')

