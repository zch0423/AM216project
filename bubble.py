#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 读入data
data = 'classified5_clean.csv'
df = pd.read_csv(data)
consumption = df['consumption']
discount = (df['discount'])
user_level = (df['user_level'])
gender = df['gender']
age = df['age']
marital = df['marital_status']
education =df['education']
city = df['city_level']
purchase = df['purchase_power']
#%%
# 定义气泡大小
size=age.rank()
n=2

# 开始做图

plt.scatter(age, education, s=size*n, alpha=0.5, cmap=cm.PuOr)
plt.xlabel('age')
plt.ylabel('education')
plt.title('consumption')
plt.show()

# plt.scatter(discount, user_level, s=size*n, alpha=0.5, cmap=cm.PuOr)
# plt.xlabel('discount')
# plt.ylabel('user_level')
# plt.title('consumption')
# plt.show()

# plt.scatter(city, purchase, s=size*n, alpha=0.5, cmap=cm.PuOr)
# plt.xlabel('city_level')
# plt.ylabel('purchase_power')
# plt.title('consumption')
# plt.show()

# %%
