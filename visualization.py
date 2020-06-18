'''
filename: visualization.py
content:
    visualization part for classification results
'''
import matplotlib.pyplot as plt
import numpy as np

def drawClassEffect(avgs, classifier="AdaBoost", classes=[4, 5, 6, 7, 8, 9, 10], save=False):
    '''
    不同分类的结果画图展示 针对某一个分类器
    stack bar
    avgs stds: array of len(classes)
    ''' 
    ind = np.arange(len(classes))
    width = 0.35
    random = np.array([1/i for i in classes])
    add_y = avgs-random  # 变为一维
    fig, ax = plt.subplots()
    random = [1/i for i in classes]  # random guess
    ax.bar(ind, random, label="Random Guess", color="orange")
    ax.bar(ind, add_y, bottom=random, label=classifier, color="steelblue")
    ax.plot(ind, random, color="sienna")
    ax.plot(ind, avgs, color="navy")
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy")
    plt.ylim((0, 0.5))
    plt.title("Accuracy of "+classifier)
    plt.legend()
    plt.xticks(ind, labels=[str(i) for i in classes])
    if save:
        plt.savefig(classifier+"_classEffect.png",dpi=600)
    plt.show()

def drawNullEffect(avgs1, avgs2, classifier="AdaBoost", classes=[4, 5, 6, 7, 8, 9, 10], save=False):
    '''
    对比柱状图
    avgs1 去除缺失值的模型
    avgs2 带缺失值训练的模型
    '''
    fig, ax = plt.subplots()
    ind = np.arange(len(classes))
    width = 0.35
    ax.bar(ind-width/2, avgs1, label="Without", color="orange", width=width)
    ax.bar(ind+width/2, avgs2, label="With", color="steelblue", width=width)
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy")
    plt.ylim((0, 0.5))
    plt.title(classifier+" Without/With Missing Values")
    plt.legend()
    plt.xticks(ind, labels=[str(i) for i in classes])
    if save:
        plt.savefig(classifier+"_missing.png", dpi=600)
    plt.show()

def drawNumEffect(avgs, stds, nums=[40, 50, 60, 70 ,80, 90, 100], save=False):
    '''
    error bar
    学习器个数对准确率的影响
    avgs: (len(nums), 4)  dtree btree rf ada
    '''
    labels = ["Decision Tree", "Bagging", "Random Forest", "AdaBoost"]
    fig, ax = plt.subplots()
    markers = ["s", ".", "v", "o"]
    for i, label in enumerate(labels):
        ax.errorbar(nums, avgs[:, i], yerr=stds[:, i], label=label,
                    alpha=0.8, capsize=4, marker=markers[i], ms=4)
    plt.xlabel("Number of Learners")
    plt.ylabel("Accuracy")
    # plt.ylim((0, 0.5))
    plt.title("Errorbar Figure of Learner Number Effect")
    plt.legend()
    if save:
        plt.savefig("learners.png", dpi=600)
    plt.show()
    

if __name__ == "__main__":
    # 测试
    # avgs = np.array([[0.40821282, 0.40874111, 0.40346531, 0.41052587],
    #                  [0.35748605, 0.35751103, 0.34113041, 0.35890672],
    #                  [0.30442125, 0.3046033, 0.29639692, 0.30783729],
    #                  [0.27963075, 0.2795665, 0.27370532, 0.28248996],
    #                  [0.25109226, 0.2509352, 0.24428871, 0.25261289],
    #                  [0.23434756, 0.23496151, 0.22692647, 0.23559334],
    #                  [0.22504176, 0.22518097, 0.21223067, 0.22560575]])
    # stds = np.array([[0.00278045, 0.00278018, 0.00354928, 0.00263594],
    #                  [0.00332105, 0.00327644, 0.00226511, 0.00301571],
    #                  [0.00253224, 0.00256523, 0.00304714, 0.00303491],
    #                  [0.00220576, 0.00228791, 0.00278641, 0.00197313],
    #                  [0.00237161, 0.00221271, 0.00358692, 0.00250307],
    #                  [0.00133697, 0.00142622, 0.00230926, 0.00251334],
    #                  [0.00267785, 0.00268569, 0.0022972, 0.00172254]])
    # avgs_n = np.array([[0.40302517, 0.40417403, 0.3988259, 0.40851198],
    #                    [0.35071729, 0.35101689, 0.33747315, 0.35666463],
    #                    [0.29816924, 0.29885262, 0.29327171, 0.30794447],
    #                    [0.27480574, 0.27508305, 0.27103976, 0.281897],
    #                    [0.24397961, 0.24423464, 0.23553893, 0.25084059],
    #                    [0.22890082, 0.22962381, 0.22322584, 0.23517499],
    #                    [0.21869723, 0.21893245, 0.20676293, 0.22446134]])

    # stds_n = np.array([[0.00230018, 0.0022015, 0.00364854, 0.00236446],
    #                    [0.00293877, 0.00286975, 0.00601432, 0.00214055],
    #                    [0.00184454, 0.00226401, 0.00272909, 0.00279257],
    #                    [0.00217648, 0.00201161, 0.00162694, 0.00176475],
    #                    [0.00256142, 0.00282895, 0.00201995, 0.00181602],
    #                    [0.00286995, 0.00302479, 0.00301089, 0.00290785],
    #                    [0.00098659, 0.00096359, 0.00134297, 0.00149681]])
    names = ["Decision Tree", "Bagging", "Random Forest", "AdaBoost"]
    # for i, name in enumerate(names):
    #     drawClassEffect(avgs[:,i], classifier=name, save=True)
    # for i, name in enumerate(names):
    #     drawNullEffect(avgs[:,i], avgs_n[:,i], classifier=name, save=True)

    # num effect
    # avgs = np.array([[0.35748606, 0.35747178, 0.33880306, 0.35926011],
    #    [0.35748606, 0.35747535, 0.34095908, 0.35872825],
    #    [0.35748606, 0.35746107, 0.33985612, 0.3581821],
    #    [0.35748606, 0.35748963, 0.34038435, 0.357843],
    #    [0.35748606, 0.35748963, 0.33996317, 0.35754316],
    #    [0.35748606, 0.35746464, 0.34042725, 0.35731113],
    #    [0.35748606, 0.35746821, 0.33836759, 0.35685067]])
    # stds = np.array([[0.00188525, 0.00184669, 0.00278472, 0.00197706],
    #    [0.00188525, 0.00185023, 0.00294802, 0.00253563],
    #    [0.00188525, 0.0018376, 0.0043283, 0.00255506],
    #    [0.00188525, 0.00187102, 0.00245267, 0.00263668],
    #    [0.00188525, 0.00185834, 0.00228898, 0.00260359],
    #    [0.00188525, 0.0018499, 0.0043138, 0.00266749],
    #    [0.00188525, 0.00184907, 0.00187952, 0.00269989]])
    # drawNumEffect(avgs, stds, save=True)
