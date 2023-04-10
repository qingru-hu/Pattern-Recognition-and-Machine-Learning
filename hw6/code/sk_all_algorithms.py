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
    ##################################################
    # TODO: 找到下面每个算法对应的sklearn模型
    ##################################################
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
    "Decision Tree": dict(criterion='log_loss', max_depth=10),
    ##################################################
    # TODO: 为下面每个模型指定超参数
    ##################################################
    # "Random Forest":
    # "Bagging":
    # "Gradient Boosting":
    # "XGBoost":
    # "Naive Bayes":
    # "Perceptron":
    # "Logistic Regression":
    # "LDA":
    # "SVM":
}

def run_method(method_name):
    # 这个函数用于调试某一个特定方法的超参数，并在训练、验证集上测试准确率
    train_results, val_results = {}, {}
    hparams = model_hparams.get(method_name, dict())
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
    model = 'Decision Tree'
    print(model)
    model_hparams_type = ['criterion', 'max_depth']
    param1 = ['log_loss', 'entropy', 'gini', ]
    param2 = [3, 5, 10, 12]
    model_hparams_values = []
    for i in param1:
        for j in param2:
            my_dict = dict()
            my_dict[model_hparams_type[0]] = i
            my_dict[model_hparams_type[1]] = j
            model_hparams_values.append(my_dict)
    print('Hyperparameter Options:')
    for item in model_hparams_values:
        print(item)
    
    vad_dat = []
    vad_mean = []
    for param in model_hparams_values:
        train_results, val_results, hparams = run_hparams(model, param)
        vad_dat.append(val_results)

    print('\nMean Validation Error')
    for i,param in enumerate(model_hparams_values):
        print(vad_dat[i])
        mean = sum(vad_dat[i].values())/len(vad_dat[i])
        print(param, mean)
        vad_mean.append(mean)
    
    ind_best = np.argmin(vad_mean)
    print('\nBest Hyperparams for ', model)
    print(model_hparams_values[ind_best])
    # train_results, val_results, hparams = run_method(model)
    # print(f'train accuracy on each dataset: {train_results}'
    #       f'\nval accuracy on each dataset: {val_results}'
    #       f'\nhparams: {hparams}')
    # report_result()