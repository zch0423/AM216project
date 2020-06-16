#%%
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from random import randint
from math import floor

def one(test1, test2):
    ll = []
    for each in test1:
        count = 0
        for sub in test2:
            if each==sub:
                count += 1
        ll.append(count)
    temp = pd.DataFrame()
    temp["1"] = test1
    temp["2"] = ll
    return temp

def main():
    test2 = [randint(1,100) for i in range(10000)]
    n = 50
    tasks = []
    with ProcessPoolExecutor(max_workers=n) as pool:
        for i in range(20000):
            test1 = list(range(1, 100))
            tasks.append(pool.submit(one, test1, test2))
    for each in tasks:
        print(each.result().iloc[1,:])


def split_n(lists, n=2):
    '''
    split an list into n pieces
    output: a list of split lists
    '''
    l = len(lists)
    output = []
    for i in range(n):
        output.append(lists[floor(i/n*l):floor((i+1)/n*l)])
    return output

orders = pd.read_csv("Data/JD_order_data.csv")
orders.set_index(["user_ID"], inplace=True, drop=False)
orders.index.set_names("hhh", inplace=True)

#%%
orders
# %%
a = orders.index.unique()
a
# %%
b = a[1:10]

# # %%
# for each in b:
#     print(orders.loc[each,["user_ID","order_ID"]])

# %%
print(orders.index)
print(orders.head())