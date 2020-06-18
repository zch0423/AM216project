'''
变量相关性热力图
'''
#%%
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

df = pd.read_csv('classified5_clean.csv')

columns = ["discount", "user_level", "plus", "gender",
           "age", "marital_status", "education", "city_level", "purchase_power"]
x = df[columns]
y = df["consumption"]

# 多项式扩充数值变量
poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=False)

x_poly = poly.fit_transform(x)
pd.DataFrame(x_poly, columns=poly.get_feature_names()).head()
# 查看热力图(颜色越深代表相关性越强)
#%%

plt.figure(figsize=(11, 9))
fig = sns.heatmap(pd.DataFrame(x_poly, columns=columns).corr(), cmap="GnBu_r")
plt.title("Correlations Between Variables")
plt.savefig("correlations.png",dpi=600)
# poly.get_feature_names()

# %%
