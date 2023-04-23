import numpy as np
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


class RLS:
    def __init__(self, gamma=1.0, kernel_type='poly', kernel_param=3.):
        self.gamma = gamma
        if kernel_type == 'poly':
            self.kernel = polynomial_kernel
        elif kernel_type == 'rbf':
            self.kernel = rbf_kernel
        self.kernel_param = kernel_param

    def fit(self, X_l, Y_l):
        self.X = X_l
        K = self.kernel(self.X, self.X, self.kernel_param)
        self.W = np.linalg.solve(K + self.gamma * len(X_l) * np.identity(len(X_l)), Y_l)
        return self

    def predict(self, X):
        K1 = self.kernel(X, self.X, self.kernel_param)
        return K1 @ self.W


class LapRLS:
    def __init__(self, gamma_A=1.0, gamma_I=1.0, n_neighbors=2, kernel_type='poly', kernel_param=3., graph_weights=9.4):
        self.gamma_A = gamma_A
        self.gamma_I = gamma_I
        self.n_neighbors = n_neighbors
        if kernel_type == 'poly':
            self.kernel = polynomial_kernel
        elif kernel_type == 'rbf':
            self.kernel = rbf_kernel
        self.kernel_param = kernel_param
        self.graph_weights = graph_weights

    def fit(self, X_l, X_u, Y_l):
        # define Y
        l = len(X_l)
        u = len(X_u)
        X = np.concatenate([X_l, X_u])
        self.X = X
        Y_u = np.zeros([u, Y_l.shape[1]])
        Y = np.concatenate([Y_l, Y_u])

        # 计算K近邻图
        W = kneighbors_graph(X, self.n_neighbors, mode='distance')
        W = np.exp(-W.todense() ** 2 / 4 * self.graph_weights)
        W = (W + W.T) / 2

        # 计算Laplacian矩阵
        D = np.diag(W.sum(axis=1))
        L = D - W

        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        K = self.kernel(X, X, self.kernel_param)
        # TODO: 将解析解带入
        self.W = np.linalg.inv(J.dot(K) + self.gamma_A*l*np.identity(l+u) + (self.gamma_I*l)/(u+l)**2*L.dot(K)).dot(Y)
        return self

    def predict(self, X):
        K1 = self.kernel(X, self.X, self.kernel_param)
        return K1.dot(self.W)


def load_dataset(name='digits'):
    if name == 'digits':
        dset = load_digits()
    elif name == 'usps':
        dset = fetch_openml('usps', version=1)
        # dset = np.load('usps.arff', allow_pickle=True)
    else:
        raise ValueError("Invalid dataset name")

    X = dset.data
    y = dset.target
    if not isinstance(y[0], int):
        y = y.astype(int)
    y = y - np.min(y)
    return X, y


def highlight_wrong_points(ax, X_t, y, y_pred, title):
    correct = y == y_pred
    incorrect = np.logical_not(correct)

    cmap = plt.cm.Set1
    norm = plt.Normalize(y.min(), y.max())

    ax.scatter(X_t[correct, 0], X_t[correct, 1], c=cmap(norm(y[correct])), edgecolor='k', marker='o')
    ax.scatter(X_t[incorrect, 0], X_t[incorrect, 1], c=cmap(norm(y[incorrect])), edgecolor='k', marker='x')

    ax.set_title(title)


def main(dataset_name):
    # 加载数据集
    X, y = load_dataset(dataset_name)

    # 预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练、验证、测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42)

    # 创建半监督数据集
    n_labeled = len(X_train) // 10
    mask = np.zeros(len(X_train), dtype=bool)
    mask[:n_labeled] = True
    np.random.shuffle(mask)

    # 监督线性回归（仅使用带有标签的样本）
    X_train_labeled = X_train[mask]
    y_train_labeled = y_train[mask]
    n_classes = len(np.unique(y))
    y_train_one_hot = np.eye(n_classes)[y_train_labeled]

    ## RLS
    # 创建并训练模型
    rls = RLS()
    rls.fit(X_train_labeled, y_train_one_hot)

    # 预测
    y_pred_rls = rls.predict(X_test)
    y_pred_rls = np.argmax(y_pred_rls, axis=1).squeeze()
    # fix the squeeze bug
    if len(y_pred_rls.shape)>1:
        y = []
        for i in y_pred_rls:
            y.append(i.item())
        y_pred_rls = np.array(y)

    # 计算准确率
    accuracy_RLS = np.mean(y_pred_rls == y_test)
    print("RLS Accuracy:", accuracy_RLS)

    ## LapRLS
    # 使用验证集选择最佳的gamma_I
    best_gamma_I = 0
    best_accuracy = 0
    for gamma_I in np.logspace(1, 6, 6):
        laprls = LapRLS(gamma_I=gamma_I)
        laprls.fit(X_train[mask], X_train[~mask], y_train_one_hot)

        y_pred_val = laprls.predict(X_val)
        y_pred_val = np.argmax(y_pred_val, axis=1).squeeze()
        # fix the squeeze bug
        if len(y_pred_val.shape)>1:
            y_pred_val = laprls.predict(X_val)
            y_pred_val = np.argmax(y_pred_val, axis=1)
            y = []
            for i in y_pred_val:
                y.append(i.item())
            y_pred_val = np.array(y)
        accuracy = np.mean(y_pred_val == y_val)
        # print(gamma_I, accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_gamma_I = gamma_I

    # 使用最佳的gamma_I重新训练模型
    laprls = LapRLS(gamma_I=best_gamma_I)
    laprls.fit(X_train[mask], X_train[~mask], y_train_one_hot)

    # 预测
    y_pred_laprls = laprls.predict(X_test)
    y_pred_laprls = np.argmax(y_pred_laprls, axis=1).squeeze()
    # fix the squeeze bug
    if len(y_pred_laprls.shape)>1:
        y_pred_laprls = laprls.predict(X_test)
        y_pred_laprls = np.argmax(y_pred_laprls, axis=1)
        y = []
        for i in y_pred_laprls:
            y.append(i.item())
        y_pred_laprls = np.array(y)

    # 计算准确率
    accuracy_manifold = np.mean(y_pred_laprls == y_test)
    print(f"LapRLS Accuracy (best gamma_I = {best_gamma_I}):", accuracy_manifold)

    methods = {
        'PCA': PCA(n_components=2),
        'LDA':LDA(),
        'MDS':MDS(n_components=2),
        'Isomap':Isomap(n_components=2),
        'LLE':LLE(n_components=2),
        't-SNE':TSNE(n_components=2)
    }

    # 绘制降维后的可视化图
    for model in ["laprls", "rls"]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        for i, (name, method) in enumerate(methods.items()):
            if name == "LDA":
                X_transformed_test = method.fit_transform(X_test, y_test)
            else:
                X_transformed_test = method.fit_transform(X_test)
            highlight_wrong_points(axs[i], X_transformed_test, y_test, eval(f"y_pred_{model}"), name)

        fig.savefig(f"{model}.pdf")


def cal_accuracy(dataset_name):
    # 加载数据集
    X, y = load_dataset(dataset_name)

    # 预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练、验证、测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42)

    # 创建半监督数据集
    n_labeled = len(X_train) // 10
    mask = np.zeros(len(X_train), dtype=bool)
    mask[:n_labeled] = True
    np.random.shuffle(mask)

    # 监督线性回归（仅使用带有标签的样本）
    X_train_labeled = X_train[mask]
    y_train_labeled = y_train[mask]
    n_classes = len(np.unique(y))
    y_train_one_hot = np.eye(n_classes)[y_train_labeled]

    ## RLS
    # 创建并训练模型
    rls = RLS()
    rls.fit(X_train_labeled, y_train_one_hot)

    # 预测
    y_pred_rls = rls.predict(X_test)
    y_pred_rls = np.argmax(y_pred_rls, axis=1).squeeze()
    if len(y_pred_rls.shape)>1:
        y = []
        for i in y_pred_rls:
            y.append(i.item())
        y_pred_rls = np.array(y)

    # 计算准确率
    accuracy_RLS = np.mean(y_pred_rls == y_test)
    print("RLS Accuracy:", accuracy_RLS)

    ## LapRLS
    # 使用验证集选择最佳的gamma_I
    # best_gamma_I = 0
    # best_accuracy = 0
    # for gamma_I in np.logspace(1, 5, 5):
    #     print(gamma_I)
    #     laprls = LapRLS(gamma_I=gamma_I)
    #     laprls.fit(X_train[mask], X_train[~mask], y_train_one_hot)

    #     y_pred_val = laprls.predict(X_val)
    #     y_pred_val = np.argmax(y_pred_val, axis=1).squeeze()
    #     if len(y_pred_val.shape)>1:
    #         # fix the bug
    #         y = []
    #         for i in y_pred_val:
    #             y.append(i.item())
    #         y_pred_val = np.array(y)
    #     accuracy = np.mean(y_pred_val == y_val)
    #     # print(gamma_I, accuracy)
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_gamma_I = gamma_I

    # 使用最佳的gamma_I重新训练模型
    # laprls = LapRLS(gamma_I=best_gamma_I)
    laprls = LapRLS(gamma_I=1e3)
    laprls.fit(X_train[mask], X_train[~mask], y_train_one_hot)

    # 预测
    y_pred_laprls = laprls.predict(X_test)
    y_pred_laprls = np.argmax(y_pred_laprls, axis=1).squeeze()
    if len(y_pred_laprls.shape)>1:
        y = []
        for i in y_pred_laprls:
            y.append(i.item())
        y_pred_laprls = np.array(y)

    # 计算准确率
    accuracy_manifold = np.mean(y_pred_laprls == y_test)
    print(f"LapRLS Accuracy (best gamma_I = {best_gamma_I}):", accuracy_manifold)
    return accuracy_RLS, accuracy_manifold


if __name__ == '__main__':
    main('digits')  # 更改数据集名称以尝试不同的数据集
    # datasets = ['digits', 'usps']
    # datasets = ['usps']
    # accuracy_RLS_all = []
    # accuracy_manifold_all = []
    # for dataset in datasets:
    #     print('#'*10, dataset, '#'*10)
    #     for i in range(5):
    #         accuracy_RLS, accuracy_manifold = cal_accuracy(dataset)
    #         accuracy_RLS_all.append(accuracy_RLS)
    #         accuracy_manifold_all.append(accuracy_manifold)
    #     print('#'*20)
    #     print('RLS: Mean Std')
    #     print(np.mean(accuracy_RLS_all), np.std(accuracy_RLS_all))
    #     print('LapRLS: Mean Std')
    #     print(np.mean(accuracy_manifold_all), np.std(accuracy_manifold_all))
