'''
filename: classification.py
content:
    classification for sales using 
    decision tree, bagging, random forest, AdaBoost
    neural networks
    introduced multi process to accelerate
'''

#%%
import numpy as np
import pandas as pd
import visualization as vis  # 绘图函数
from sklearn import ensemble, tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from concurrent.futures import ProcessPoolExecutor  # 多进程

#%%
# 进程个数  根据电脑cpu调整
max_workers = 10
# hyperparameters
num = 50  # number of ensemble models
leaf = 5  # min sample leaf
depth = 5  # max depth

#%%
def decisionTree(X_train, y_train, X_test, y_test):
    '''
    fitting using decision tree model
    INPUTS: X_train y_train X_test y_test
    RETURNS: accuracy 
    '''
    dtree = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
    dtree.fit(X_train, y_train)
    yhat = dtree.predict(X_test)
    return accuracy_score(y_test, yhat)

def baggingTree(X_train, y_train, X_test, y_test):
    '''
    fitting using baggingTree
    INPUTS: X_train y_train X_test y_test
    RETURNS: accuracy
    '''
    btree = ensemble.BaggingClassifier(n_estimators=num, base_estimator=tree.DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=leaf))
    btree.fit(X_train, y_train)
    y_hat = btree.predict(X_test)
    return accuracy_score(y_test, y_hat)

def randomForest(X_train, y_train, X_test, y_test):
    '''
    fitting using random forest
    INPUTS: X_train y_train X_test y_test
    RETURNS: accuracy
    '''
    rf = ensemble.RandomForestClassifier(n_estimators=num,
                                         max_depth=depth, min_samples_leaf=leaf)
    rf.fit(X_train, y_train)
    yhat = rf.predict(X_test)
    return accuracy_score(y_test, yhat)

def adaBoost(X_train, y_train, X_test, y_test):
    ada = ensemble.AdaBoostClassifier(n_estimators=num, base_estimator=tree.DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=leaf))
    ada.fit(X_train, y_train)
    yhat = ada.predict(X_test)
    return accuracy_score(y_test, yhat)

def oneTrain(X_train, y_train, X_test, y_test):
    '''
    training use four models
    RETURNS: accuracy--dtree, bagging, random forest, adaBoost
    '''
    # decision tree
    dtree = decisionTree(X_train, y_train, X_test, y_test)
    # bagging
    btree = baggingTree(X_train, y_train, X_test, y_test)
    # random forest
    rf = randomForest(X_train, y_train, X_test, y_test)
    # adaBoost
    ada = adaBoost(X_train, y_train, X_test, y_test)
    return (dtree, btree, rf, ada)

def training(X, y, max_workers=6):
    '''
    function for one training process
    using decision tree, bagging, random forest, adaBoost
    INPUTS: X, y, number of processes max_workers default 6
    RETURNS: array of mean accuracy std--[dtree, bagging, rf, ada]
    '''
    tasks = []
    # 10 fold
    kf = KFold(n_splits=10)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for trainIdx, testIdx in kf.split(X):
            X_train, y_train = X[trainIdx], y[trainIdx]
            X_test, y_test = X[testIdx], y[testIdx]
            tasks.append(pool.submit(oneTrain, X_train, y_train, X_test, y_test))
    # [[dtree, bagging, rf, ada],[],[]]
    accuracy = np.array([task.result() for task in tasks])
    avg_accuracy = np.mean(accuracy, axis=0)
    std_accuracy = np.std(accuracy, axis=0, ddof=1)
    return avg_accuracy, std_accuracy

def classEffect(classes = [4, 5, 6, 7, 8, 9, 10], null=False):
    '''
    测试不同分类预测效果
    '''
    avgs = []
    stds = []
    for n in classes:
        if null:
            infile = "classified%d.csv" % n
        else:
            infile = "classified%d_clean.csv"%n
        print("Learning", infile)
        data = pd.read_csv(infile)
        data = shuffle(data)  # 打乱
        X = data.iloc[:, 2:].to_numpy()  # consumption
        # discount,user_level,plus,gender,age,marital_status,education,city_level,purchase_power
        y = data.iloc[:, 1].to_numpy()
        avg_accuracy, std_accuracy = training(X, y, max_workers=max_workers)
        avgs.append(avg_accuracy)
        stds.append(std_accuracy)
    return np.array(avgs), np.array(stds)

def numEffect(X, y, num_low=40, num_high=110, step=10):
    '''
    adaBoost 学习器个数
    [num_low, num_high)
    '''
    avgs = []
    stds = []
    for i in range(num_low, num_high, step):
        print("Num:", i)
        global num
        num = i
        avg, std = training(X, y, max_workers=max_workers)
        avgs.append(avg)
        stds.append(std)
    return np.array(avgs), np.array(stds)

# 神经网络部分
def oneNetwork(X_train, y_train, X_test, y_test, activation="relu", layers=(20, )):
    '''
    神经网络一次训练子函数,返回训练集上准确度和训练集上准确度
    '''
    m = MLPClassifier(activation=activation, hidden_layer_sizes=layers)
    m.fit(X_train, y_train)
    score = accuracy_score(y_test, m.predict(X_test))
    return score

def networkTraining(X, y, activation="relu", layers=(20, )):
    '''
    神经网络一次训练子函数
    '''
    tasks = []
    # 10 fold
    kf = KFold(n_splits=10)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for trainIdx, testIdx in kf.split(X):
            X_train, y_train = X[trainIdx], y[trainIdx]
            X_test, y_test = X[testIdx], y[testIdx]
            tasks.append(pool.submit(
                oneNetwork, X_train, y_train, X_test, y_test, activation, layers))
    accuracy = np.array([task.result() for task in tasks])
    avg_accuracy = np.mean(accuracy, axis=0)
    std_accuracy = np.std(accuracy, axis=0, ddof=1)
    return avg_accuracy, std_accuracy
    
def nodeLayerEffect(X, y, max_layer=5, node_low=1, node_high=6, step=1, activation=""):
    '''
    不同节点的预测准确率
    RETURNS: [[1层 ],[2层],...]
    '''
    avgs = []
    for layer in range(1, max_layer+1):
        temp = []
        for i in range(node_low, node_high, step):
            layers = [i for i in range(layer)]
            print("Layers:", layers)
            avg, std = networkTraining(X, y, activation=activation, layers=layers)
            temp.append(avg)
        avgs.append(temp)
        return np.array(avgs)

#%%
# 取消注释即可运行
print("取消注释运行")
# avgs, stds = classEffect(null=False)
# avgs_m, stds_m = classEffect(null=True)
# %%
# 不同分类数  是否包含缺失值模型比较
# for i, name in enumerate(["Decision Tree", "Bagging", "Random Forest", "AdaBoost"]):
#     vis.drawClassEffect(avgs[:,i], classifier=name)
# for i, name in enumerate(names):
#     drawNullEffect(avgs[:,i], avgs_m[:,i], classifier=name, save=False)

#%%
data = pd.read_csv("classified5_clean.csv")
data = shuffle(data)  # 打乱
X = data.iloc[:, 2:].to_numpy()  # consumption
# discount,user_level,plus,gender,age,marital_status,education,city_level,purchase_power
y = data.iloc[:, 1].to_numpy()
# 学习器个数 leaners
# avgs_l, stds_l = numEffect(X, y, num_low=10, num_high=30, step=10)
# vis.drawNumEffect(avgs_l, stds_l)
num = 50  # reset

# %%
# 神经网络
# avgs_nn = nodeLayerEffect(X, y)


# %%
