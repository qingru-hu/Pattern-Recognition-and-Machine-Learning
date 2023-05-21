from MPM import MPM
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from dataloader import dataLoader_cancer, dataLoader_diabetes, dataLoader_sonar
import numpy as np
from train import train

path_cancer = './dataset/breast-cancer-wisconsin.txt'
path_diaveres = './dataset/diabetes.csv'
path_sonar = './dataset/sonar_csv.csv'

X_can, y_can = dataLoader_cancer(path_cancer)
X_dia, y_dia = dataLoader_diabetes(path_diaveres)
X_son, y_son = dataLoader_sonar(path_sonar)

model_MPM = MPM()
model_LDA = LinearDiscriminantAnalysis()
model_LGS = LogisticRegression(max_iter=114514)
model_SVM = SVC()

train(X_can, y_can, model_MPM, model_LDA, model_LGS, model_SVM, 'breast cancer')
train(X_dia, y_dia, model_MPM, model_LDA, model_LGS, model_SVM, 'diabetes')
train(X_son, y_son, model_MPM, model_LDA, model_LGS, model_SVM, 'sonar')