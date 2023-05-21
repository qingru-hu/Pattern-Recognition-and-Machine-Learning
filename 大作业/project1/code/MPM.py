import numpy as np
import pandas as pd
from time import time
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


## define functions and classes
# define the functions to load data
def load_cancer(path):
    df = pd.read_table(path, sep='\t', header=None, na_values=['?'])
    df.fillna(df.mean(), inplace=True)
    
    X = df.iloc[:, 1:10].to_numpy()
    y = df.iloc[:, 10].to_numpy()
    return X, y

def load_diabetes(path):
    df = pd.read_csv(path)
    df.fillna(df.mean(), inplace=True)
    X = df.iloc[:, :8].to_numpy()
    y = df.iloc[:, 8].to_numpy()
    return X, y

def load_sonar(path):
    df = pd.read_csv(path)
    df['Class'] = df['Class'].replace(['Rock'], 1)
    df['Class'] = df['Class'].replace(['Mine'], 0)
    df.fillna(df.mean(), inplace=True)
    X = df.iloc[:, :60].to_numpy()
    y = df.iloc[:, 60].to_numpy()
    return X, y

# define the MPM model class
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

# define the train function
def train(X, y, model_MPM, model_LDA, model_LGS, model_SVM, dataset:str):
    MPM_err = 0
    MPM_acc = 0
    LDA_acc = 0
    LGS_acc = 0
    SVM_acc = 0
    MPM_t = 0
    LDA_t = 0
    LGS_t = 0
    SVM_t = 0

    for i in range(0, 10):
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state = i + 1,shuffle = True)

        # fit and timing
        t1 = time()
        model_MPM.fit(X_train, y_train)
        t2 = time()
        MPM_t += t2 - t1
        
        model_LDA.fit(X_train, y_train)
        t3 = time()
        LDA_t += t3 - t2
        
        model_LGS.fit(X_train, y_train)
        t4 = time()
        LGS_t += t4 - t3
        
        model_SVM.fit(X_train, y_train)
        t5 = time()
        SVM_t += (t5 - t4)

        # predict
        y_MPM = model_MPM.predict(X_test)
        y_LDA = model_LDA.predict(X_test)
        y_LGS = model_LGS.predict(X_test)
        y_SVM = model_SVM.predict(X_test)

        # calculate accuracy and error
        MPM_acc += np.sum(y_MPM == y_test) / y_test.shape[0]
        LDA_acc += np.sum(y_LDA == y_test) / y_test.shape[0]
        LGS_acc += np.sum(y_LGS == y_test) / y_test.shape[0]
        SVM_acc += np.sum(y_SVM == y_test) / y_test.shape[0]

        MPM_err += model_MPM.e

    MPM_acc /= 10
    MPM_err /= 10
    LDA_acc /= 10
    LGS_acc /= 10
    SVM_acc /= 10
    MPM_t /= 10
    LDA_t /= 10
    LGS_t /= 10
    SVM_t /= 10

    print(f'Dataset: {dataset}')
    print(f'LDA: accuracy = {LDA_acc:.3f}, time = {LDA_t:.3f} s')
    print(f'LGS: accuracy = {LGS_acc:.3f}, time = {LGS_t:.3f} s')
    print(f'SVM: accuracy = {SVM_acc:.3f}, time = {SVM_t:.3f} s')
    print(f'MPM: accuracy = {MPM_acc:.3f}, time = {MPM_t:.3f} s, error = {MPM_err:.3f}')


## train different models on different datasets
if __name__ == '__main__':
    # load data
    path_can = 'data/breast-cancer-wisconsin.txt'
    path_dia = 'data/diabetes.csv'
    path_son = 'data/sonar_csv.csv'

    X_can, y_can = load_cancer(path_can)
    X_dia, y_dia = load_diabetes(path_dia)
    X_son, y_son = load_sonar(path_son)
    
    # load model
    model_MPM = MPM()
    model_LDA = LinearDiscriminantAnalysis()
    model_LGS = LogisticRegression(max_iter=114514)
    model_SVM = SVC()

    # train
    train(X_can, y_can, model_MPM, model_LDA, model_LGS, model_SVM, 'breast cancer')
    train(X_dia, y_dia, model_MPM, model_LDA, model_LGS, model_SVM, 'diabetes')
    train(X_son, y_son, model_MPM, model_LDA, model_LGS, model_SVM, 'sonar')