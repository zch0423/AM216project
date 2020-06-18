'''
filename: dataprocess.py
content: process data to generate tables with multi process 
        you should adjust the value of process_num according to machine power for
        multiUserOrder(user, order, process_num=6, outfile="userorder.csv") 
'''

#%%
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from math import floor

#%%
def read_tables(dirPath = "Data"):
    '''
    input: directory path
    output: tables of clicks order sku user
    '''
    clicks = pd.read_csv(dirPath+"/JD_click_data.csv")
    order = pd.read_csv(dirPath+"/JD_order_data.csv")
    sku = pd.read_csv(dirPath+"/JD_sku_data.csv")
    user = pd.read_csv(dirPath+"/JD_user_data.csv")
    # drop unused columns
    order_labels = ["order_date", "order_time", "promise",
                     "direct_discount_per_unit", "quantity_discount_per_unit",
                     "bundle_discount_per_unit", "coupon_discount_per_unit",
                     "gift_item", "dc_ori", "dc_des"]
    user_labels = ["first_order_month"]
    sku_labels = ["activate_date", "deactivate_date"]
    order = order.drop(labels=order_labels,axis=1)
    user = user.drop(labels=user_labels, axis=1)
    sku = sku.drop(labels=sku_labels, axis=1)
    order = order[order["final_unit_price"]>0]  # delete unusual price
    return clicks, order, sku, user

def getUserOrderTable(user, order):
    '''
    warning: it may take several hours  O(n^2)
    merge user and order
    add up all consumption of one user
    output: a merged table
    '''
    consumption = []
    discount = []
    uniqueID = order["user_ID"].unique()
    for each in uniqueID:
        temp = order[order["user_ID"]==each]
        temp_consumption = 0
        temp_discount = 0
        print(each)  # test
        for _, row in temp.iterrows():
            q = row["quantity"]
            price1 = row["original_unit_price"]
            price2 = row["final_unit_price"]
            temp_discount += q*(price1-price2)
            temp_consumption += q*price2
        consumption.append(temp_consumption)
        discount.append(temp_discount)
    leftTable = pd.DataFrame()
    leftTable["user_ID"] = uniqueID
    leftTable["consumption"] = consumption
    leftTable["discount"] = discount
    merged = pd.merge(leftTable, user, how="left", on="user_ID")
    return merged
    
# %%
def oneProcess(uniqueUsers, order):
    '''
    function for single process dealing with part of the unique users
    '''
    consumption = []
    discount = []
    for each in uniqueUsers:
        temp = order[order["user_ID"] == each]
        temp_consumption = 0
        temp_discount = 0
        # print(each)  # test
        for _, row in temp.iterrows():
            q = row["quantity"]
            price1 = row["original_unit_price"]
            price2 = row["final_unit_price"]
            temp_discount += q*(price1-price2)
            temp_consumption += q*price2
        consumption.append(temp_consumption)
        discount.append(temp_discount)
    leftTable = pd.DataFrame()
    leftTable["user_ID"] = uniqueUsers
    leftTable["consumption"] = consumption
    leftTable["discount"] = discount 
    return leftTable

def split_n_generator(lists, n=2):
    '''
    a generator
    split an list into n pieces
    '''
    l = len(lists)
    for i in range(n):
        yield lists[floor(i/n*l):floor((i+1)/n*l)]


def multiUserOrder(user, order, process_num=6, outfile="userorder.csv"):
    '''
    apply multiprocess to accelerate
    process_num can change according to machine computing power
    output: a csv file into outfile
    '''
    usefulLabels = ["user_ID", "quantity",
                    "original_unit_price", "final_unit_price"]
    order = order[usefulLabels]
    uniqueUsers = user["user_ID"].unique()
    order.set_index("user_ID", inplace=True, drop=False)
    order.index.set_names("index", inplace=True)  # avoid ambiguity when concat
    tasks = []
    with ProcessPoolExecutor(max_workers=process_num) as pool:
        for split_users in split_n_generator(uniqueUsers, n=process_num):
            tasks.append(pool.submit(oneProcess, split_users, order))
    temp = pd.concat([task.result() for task in tasks])
    merged = pd.merge(temp, user, how="left", on="user_ID")
    merged.to_csv(outfile, index=False, header=True)

#%%
# 数据预处理
def process_data(n_cons=10, n_dis=5, path="userorder.csv"):
    '''
    将data的consumption 分n_cons类 改成数字1-n
    discount 分成n_dis类
    将其他非数值数据都数值化
    '''
    # user_ID,consumption,discount,user_level,
    # plus,gender,age,marital_status,education,city_level,purchase_power
    data = pd.read_csv(path)
    data = data[data["consumption"] > 1e-5]
    print("Labelling consumption")
    consumption = data["consumption"]
    # 分割成n份
    cuts, bins = pd.qcut(consumption, n_cons, retbins=True)
    for i, temp in consumption.iteritems():
        for j in range(1, n_cons+1):
            if temp <= bins[j]:
                consumption[i] = j
                break
    consumption = consumption.astype(int)
    data["consumption"] = consumption

    print("Labelling discount")
    discount = data["discount"]
    cuts, bins = pd.qcut(discount, n_dis, retbins=True)
    for i, temp in discount.iteritems():
        for j in range(1, n_dis+1):
            if temp <= bins[j]:
                discount[i] = j
                break
    discount = discount.astype(int)
    data["discount"] = discount

    # 将其余转化为数值
    gender_map = {"U": -1, "F": 1, "M": 0}
    marital_map = {"U": -1, "M": 1, "S": 0}
    age_map = {"U": -1,
               "<=15": 1,
               "16-25": 2,
               "26-35": 3,
               "36-45": 4,
               "46-55": 5,
               ">=56": 6}
    print("Labelling gender")
    data["gender"] = data["gender"].map(gender_map)
    print("Labelling age")
    data["age"] = data["age"].map(age_map)
    print("Labelling marital status")
    data["marital_status"] = data["marital_status"].map(marital_map)
    data.to_csv("classified%d.csv"%n_cons, index=0)
    return data

def deleteNull(infile="classified10.csv"):
    data = pd.read_csv(infile)
    data = data[~data["gender"].isin([-1])]
    data = data[~data["age"].isin([-1])]
    data = data[~data["marital_status"].isin([-1])]
    data = data[~data["education"].isin([-1])]
    data = data[~data["city_level"].isin([-1])]
    data = data[~data["purchase_power"].isin([-1])]
    outfile = infile[:-4]+"_clean.csv"
    data.to_csv(outfile, index=False)

#%%
if __name__ == "__main__":
    # clicks, order, sku, user = read_tables()
    print("data process")
    # multiUserOrder(user, order, process_num=6, outfile="userorder.csv")
    for n in [4, 5, 6, 7, 8, 9, 10]:
        # 分类
        print("分类:", n)
        process_data(n_cons=n)  # 预处理
        deleteNull(infile="classified%d.csv"%n)
