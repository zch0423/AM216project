'''
气泡图
'''
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("classified5_clean.csv")

#%%
def drawBubble(xlabel, ylabel, zlabel="consumption"):
    '''
    气泡图
    '''
    sns.set(style="whitegrid")  # style
    x = data[xlabel] 
    y = data[ylabel]
    z = data[zlabel]
    cm = plt.cm.get_cmap('RdYlBu')
    fig, ax = plt.subplots(figsize=(12, 10))
    #注意s离散化的方法，因为需要通过点的大小来直观感受其所表示的数值大小
    #我所使用的是当前点的数值减去集合中的最小值后+0.1再*1000
    #参数是X轴数据、Y轴数据、各个点的大小、各个点的颜色
    bubble = ax.scatter(x, y, s=(z - np.min(z) + 0.1) * 1000,
                        c=z, cmap=cm, linewidth=0.5, alpha=0.5)
    ax.grid()
    fig.colorbar(bubble)
    ax.set_xlabel('people of cities', fontsize=15)  # X轴标签
    ax.set_ylabel('price of something', fontsize=15)  # Y轴标签
    plt.show()
#%%
drawBubble("age", "education")
# %%
