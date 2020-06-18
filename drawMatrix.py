'''
绘制混淆矩阵
'''
#%%
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
#%%
leaf = 5  # min sample leaf
depth = 10  # max dept
data = pd.read_csv("classified5_clean.csv")
data = shuffle(data)  # 打乱
X = data.iloc[:, 2:].to_numpy()  # consumption
# discount,user_level,plus,gender,age,marital_status,education,city_level,purchase_power
y = data.iloc[:, 1].to_numpy()
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

tree=DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
tree.fit(X_train, y_train)
yhat = tree.predict(X_test)


m = MLPClassifier(verbose=True, activation="relu", hidden_layer_sizes=(10, ))
m.fit(X_train, y_train)
yhat_nn = m.predict(X_test)

#%%
def cmPlot(y, yp, classifier="Decision Tree", save=False, outfile="confusion.png"):
    cm = confusion_matrix(y, yp, normalize="pred") #混淆矩阵
    fig, ax = plt.subplots()
    heat = ax.matshow(cm, cmap=plt.cm.GnBu)
    
    # for x in range(len(cm)):
    #     for y in range(len(cm)):
    #         plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True Label') #坐标轴标签
    plt.xlabel('Predicted Label') #坐标轴标签
    plt.xticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
    plt.yticks(ticks=[0,1,2,3,4], labels=[1,2,3,4,5])
    plt.title("Confusion Matrix of "+classifier)
    # plt.colorbar()
    fig.colorbar(heat, ax=ax)
    if save:
        plt.savefig(outfile, dpi=600)
    plt.show()
#%%
cmPlot(y_test, yhat, save=True)

# %%
cmPlot(y_test, yhat_nn, classifier="Networks", save=True, outfile="confusionnn.png")

# %%
