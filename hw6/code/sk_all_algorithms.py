from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据集并进行train/val/test划分
datasets = {
    "iris": load_iris(),
    "breast_cancer": load_breast_cancer(),
    "usps": fetch_openml('usps', version=3)
}

X_train_sets, X_val_sets, X_test_sets, y_train_sets, y_val_sets, y_test_sets = {}, {}, {}, {}, {}, {}
scaler = StandardScaler()

print("Loading datasets")
for name, data in datasets.items():
    print(f"=> Loading {name}")
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_sets[name], X_val_sets[name], X_test_sets[name], y_train_sets[name], y_val_sets[name], y_test_sets[name] = X_train_scaled, X_val_scaled, X_test_scaled, y_train.astype(int), y_val.astype(int), y_test.astype(int)
    print("=> Done!")
print("Dataset loading complete.")

# 这里我们以Decision Tree为例，定义了模型以及其超参数（criterion选什么以及max_depth怎么取），请大家完成其他模型的定义以及超参数设置
models = {
    "Decision Tree": DecisionTreeClassifier,  # criterion, max_depth
    "Random Forest": RandomForestClassifier, # criterion, n_estimators
    "Bagging": BaggingClassifier, # n_estimators
    "Gradient Boosting": GradientBoostingClassifier, # n_estimators, learning_rate
    # "XGBoost":
    "Naive Bayes": GaussianNB, # var_smoothing
    "Perceptron": Perceptron, # penalty, alpha
    "Logistic Regression": LogisticRegression, # penalty, C
    "LDA": LinearDiscriminantAnalysis,
    "SVM": SVC, # kernel, C
}

model_hparams_best = {
    "Decision Tree": dict(criterion='gini', max_depth=5),
    "Random Forest": dict(n_estimators=50, criterion='entropy'),
    "Bagging": dict(n_estimators=20),
    "Gradient Boosting": dict(learning_rate=0.1, n_estimators=100),
    # "XGBoost":
    "Naive Bayes":dict(var_smoothing=1e-9),
    "Perceptron":dict(alpha=0.001, penalty='l1'),
    "Logistic Regression": dict(C=1, penalty='l2'),
    "LDA":dict(),
    "SVM":dict(kernel='rbf', C=1)
}

def run_method(method_name):
    # 这个函数用于调试某一个特定方法的超参数，并在训练、验证集上测试准确率
    train_results, val_results = {}, {}
    hparams = model_hparams_best.get(method_name, dict())
    print(hparams)
    for name, X_train, X_val, y_train, y_val in zip(
            datasets.keys(), X_train_sets.values(), X_val_sets.values(), y_train_sets.values(), y_val_sets.values()
    ):
        model_fn = models[method_name]
        model = model_fn(**hparams)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        train_results[name] = train_score
        val_results[name] = val_score
    return train_results, val_results, hparams

def run_hparams(method_name, param):
    # 这个函数用于调试某一个特定方法的超参数，并在训练、验证集上测试准确率
    train_results, val_results = {}, {}
    hparams = param #model_hparams.get(method_name, dict())
    for name, X_train, X_val, y_train, y_val in zip(
            datasets.keys(), X_train_sets.values(), X_val_sets.values(), y_train_sets.values(), y_val_sets.values()
    ):
        model_fn = models[method_name]
        model = model_fn(**hparams)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        train_results[name] = train_score
        val_results[name] = val_score
    return train_results, val_results, hparams

def report_result(runs = 5):
    # 这个函数将每个算法都运行多次，用于减少随机误差。并汇总了每个算法在所有数据集的测试集上的结果，以柱状图形式呈现最终表现。
    results = []
    for name, X_train, X_test, y_train, y_test in zip(
        datasets.keys(), X_train_sets.values(), X_test_sets.values(), y_train_sets.values(), y_test_sets.values()
    ):
        row = {"Dataset": name}

        for model_name, model_fn in models.items():
            score = 0
            for _ in range(runs):
                model = model_fn(**model_hparams_best.get(model_name, dict()))
                model.fit(X_train, y_train)
                score += model.score(X_test, y_test)
            row[model_name] = score / runs
        results.append(row)

    df = pd.DataFrame(results)
    df_melt = pd.melt(df, id_vars=["Dataset"], var_name="Algorithm", value_name="Accuracy")

    sns.catplot(x="Dataset", y="Accuracy", hue="Algorithm", data=df_melt, kind="bar", height=6, aspect=1.5)
    plt.ylim(0.8, 1.0)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    datanames = ['iris', 'breast_cancer', 'usps']
    
    # model = 'Decision Tree'
    # model_hparams_type = ['max_depth', 'criterion',]
    # param1 = [3, 5, 10, 12]
    # param2 = ['entropy', 'gini', 'log_loss']
    
    # model = 'Random Forest'
    # model_hparams_type = ['n_estimators', 'criterion',]
    # param1 = [20, 50, 100, 200]
    # param2 = ['entropy', 'gini', 'log_loss']
    
    # model = 'Bagging'
    # model_hparams_type = ['n_estimators']
    # param1 = [2, 5, 10, 20]
    
    # model = 'Gradient Boosting'
    # model_hparams_type = ['n_estimators', 'learning_rate',]
    # param1 = [10, 20, 50, 100, 200]
    # param2 = [1e-3, 1e-2, 1e-1]
    
    # model = 'Naive Bayes'
    # model_hparams_type = ['var_smoothing']
    # param1 = [1e-9, 1e-7, 1e-3, 1e-1]
    
    # model = 'Perceptron'
    # model_hparams_type = ['alpha', 'penalty',]
    # param1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # param2 = [None, 'l1', 'l2']

    # model = 'Logistic Regression'
    # model_hparams_type = ['C', 'penalty']#, 'solver']
    # param1 = [0.1, 1, 10]
    # param2 = [None, 'l2']
    # param3 = ['lbfgs', 'liblinear', 'lbfgs']
    
    # model = 'LDA'
    # vad_dat = []
    # vad_mean = []
    # train_results, val_results, hparams = run_method(model)
    # vad_dat= val_results
    # print(vad_dat)
    # print(sum(vad_dat.values())/len(vad_dat))
    
    # model = 'SVM'
    # model_hparams_type = ['C', 'kernel']
    # param1 = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    # param2 = ['linear', 'poly', 'rbf', 'sigmoid']

    
    # print(model)
    # model_hparams_values = []
    # if len(model_hparams_type)==1:
    #     for i in param1:
    #         my_dict = dict()
    #         my_dict[model_hparams_type[0]] = i
    #         model_hparams_values.append(my_dict)
    # if len(model_hparams_type)==3:
    #     for i in param1:
    #         for j in param2:
    #             for k in param3:
    #                 my_dict = dict()
    #                 my_dict[model_hparams_type[0]] = i
    #                 my_dict[model_hparams_type[1]] = j
    #                 my_dict[model_hparams_type[2]] = k
    #                 model_hparams_values.append(my_dict)
    # else:
    #     for i in param1:
    #         for j in param2:
    #             my_dict = dict()
    #             my_dict[model_hparams_type[0]] = i
    #             my_dict[model_hparams_type[1]] = j
    #             model_hparams_values.append(my_dict)
    # print('Hyperparameter Options:')
    # for item in model_hparams_values:
    #     print(item)
    
    # vad_dat = []
    # vad_mean = []
    # for param in model_hparams_values:
    #     train_results, val_results, hparams = run_hparams(model, param)
    #     vad_dat.append(val_results)

    # print('\nMean Validation Error')
    # for i,param in enumerate(model_hparams_values):
    #     # print(vad_dat[i])
    #     mean = sum(vad_dat[i].values())/len(vad_dat[i])
    #     # print(param, mean)
    #     vad_mean.append(mean)
   
    # if len(model_hparams_type)==1:
    #     for i in range(4):
    #         row = str(param1[i])
    #         for k in range(3):
    #             row += ' & '
    #             row += str(round(vad_dat[i][datanames[k]], 3))
    #         row += ' \\\ \hline'
    #         print(row)
    # else:
    #     for i in range(len(param1)):
    #         row = str(param1[i])
    #         for j in range(len(param2)):
    #             for k in range(len(datanames)):
    #                 row += ' & '
    #                 row += str(round(vad_dat[j*3 + i][datanames[k]], 3))
    #         row += ' \\\ \hline'
    #         print(row)
    
    # ind_best = np.argmax(vad_mean)
    # print('\nBest Hyperparams for ', model)
    # print(model_hparams_values[ind_best])
    # train_results, val_results, hparams = run_method(model)
    # print(f'train accuracy on each dataset: {train_results}'
    #       f'\nval accuracy on each dataset: {val_results}'
    #       f'\nhparams: {hparams}')
    report_result()