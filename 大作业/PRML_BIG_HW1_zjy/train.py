import numpy as np
from sklearn.model_selection import train_test_split
from MPM import MPM
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from time import time

def train(X, y, model_MPM:MPM, model_LDA:LinearDiscriminantAnalysis, model_LGS:LogisticRegression, model_SVM:SVC, name:str):

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
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1,random_state = i + 1,shuffle = True)

        t1 = time()
        model_MPM.fit(X_train, y_train)
        t2 = time()
        model_LDA.fit(X_train, y_train)
        t3 = time()
        model_LGS.fit(X_train, y_train)
        t4 = time()
        model_SVM.fit(X_train, y_train)
        t5 = time()

        y_MPM = model_MPM.predict(X_test)
        y_LDA = model_LDA.predict(X_test)
        y_LGS = model_LGS.predict(X_test)
        y_SVM = model_SVM.predict(X_test)

        MPM_acc += np.sum(y_MPM == y_test) / y_test.shape[0]
        LDA_acc += np.sum(y_LDA == y_test) / y_test.shape[0]
        LGS_acc += np.sum(y_LGS == y_test) / y_test.shape[0]
        SVM_acc += np.sum(y_SVM == y_test) / y_test.shape[0]

        MPM_t = MPM_t + (t2 - t1)
        LDA_t = LDA_t + (t3 - t2)
        LGS_t = LGS_t + (t4 - t3)
        SVM_t = SVM_t + (t5 - t4)

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

    print(name)
    print('LDA: ' + ' accuracy: ', LDA_acc, 'time: ', LDA_t)
    print('LGS: ' + ' accuracy: ', LGS_acc, 'time: ', LGS_t)
    print('SVM: ' + ' accuracy: ', SVM_acc, 'time: ', SVM_t)
    print('MPM: ' + ' accuracy: ', MPM_acc, 'time: ', MPM_t)
    print('MPM guaranteed error: ', MPM_err)