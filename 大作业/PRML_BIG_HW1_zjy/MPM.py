import cvxpy as cp
import numpy as np

class MPM():
    def __init__(self):
        pass

    def optimize(self):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == 0]

        cov1 = np.cov(X1.T)
        cov2 = np.cov(X2.T)

        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)

        dim = mean1.shape[0]

        Q1 = np.linalg.cholesky(cov1)
        Q2 = np.linalg.cholesky(cov2)

        w = cp.Variable(dim)

        F0 = (mean1 - mean2)
        F = F0.T

        prob = cp.Problem(cp.Minimize(cp.norm(Q1.T@w) + cp.norm(Q2.T@w)),
                  [F @ w == 1])
        prob.solve()
        w_ = w.value
        k_1 = 1 / (np.sqrt(w_.T @ cov1 @ w_) + np.sqrt(w_.T @ cov2 @ w_))
        b = w_.T @ mean1 - k_1 * np.sqrt(w_.T @ cov2 @ w_)

        self.w = w_
        self.b = b
        self.e = 1 / (k_1**2 + 1)

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.X = X
        self.y = y
        self.optimize()

    def predict(self, X):
        y = np.matmul(self.w, X.T)
        y = y - self.b
        result = y
        result[y >= 0] = 1
        result[y < 0] = 0
        
        return result

