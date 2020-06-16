'''
filename: classification.py
content:
    classification for sales using 
    decision tree, bagging, random forest, AdaBoost
    introduced multi process to accelerate
'''

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from concurrent.futures import ProcessPoolExecutor  # 多进程

#%%
data = pd.read_csv("classified.csv")
data = shuffle(data)  # 打乱
X = data.iloc[:, 2:].to_numpy()  # consumption
# discount,user_level,plus,gender,age,marital_status,education,city_level,purchase_power
y = data.iloc[:, 1].to_numpy()

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

if __name__ == "__main__":
    X = np.zeros((6, 5))
    X[0] = [np.nan, 1, 2, 3, 4]
    X[1] = [1, 2, 3, 4, np.nan]
    X[2] = [np.nan, 5, 4, 3, 2]
    X[3] = [np.nan, 5, 4, 4, 1]
    X[4] = [4, np.nan, 4, 4, 1]
    X[5] = [np.nan, 2, 3, 4, 5]
    y = np.array([1, 1, 0,0,0,1 ])
    print(X)
    print(baggingTree(X, y, X, y))
    # avg_accuracy, std_accuracy = training(X, y, max_workers=8)
    # print(avg_accuracy, std_accuracy)
