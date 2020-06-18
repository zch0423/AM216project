#%%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import csv
import numpy as np
# 读入data
data = 'classified5_clean.csv'
df = pd.read_csv(data)
consumption = df['consumption']
discount = df['discount']
user_level = df['user_level']
gender = df['gender']
age = df['age']
marital = df['marital_status']
education = df['education']
city = df['city_level']
purchase = df['purchase_power']

temp1 = df[df["gender"]==0]
temp2 = df[df["gender"]!=0]
y1 = temp1["consumption"].to_numpy()
Y1 = temp2["consumption"].to_numpy()
x1 = temp1["discount"].to_numpy()
X1 = temp2["discount"].to_numpy()
x2 = temp1["user_level"].to_numpy()
X2 = temp2["user_level"].to_numpy()
x3 = temp1["education"].to_numpy()
X3 = temp2["education"].to_numpy()
x4 = temp1["city_level"].to_numpy()
X4 = temp2["city_level"].to_numpy()
x5 = temp1["purchase_power"].to_numpy()
X5 = temp2["purchase_power"].to_numpy()
#%%

#%%

x=np.column_stack((x1,x2,x3,x4,x5))
x_n=sm.add_constant(x)
model=sm.OLS(y1, x_n)
results=model.fit()
print('Parameters:', results.params)
params1 = results.params

#%%
X=np.column_stack((X1,X2,X3,X4,X5))
X_n=sm.add_constant(X)
Model=sm.OLS(Y1, X_n)
Results=model.fit()

print('Parameters:', Results.params)
params2 = Results.params


#%%

x=np.column_stack((x1,x2,x3,x4,x5))
x_n=sm.add_constant(x)
model=sm.OLS(y1, x_n)
results=model.fit()

print('Parameters:', results.params)
params3 = results.params

#%%
X=np.column_stack((X1,X2,X3,X4,X5))
X_n=sm.add_constant(X)
Model=sm.OLS(Y1, X_n)
Results=model.fit()

print('Parameters:', Results.params)
params4= Results.params

feature = ['user_level','age','education','city_level','purchase_power']
f=open('params.csv','w',encoding='utf-8')
csv_writer=csv.writer(f)
csv_writer.writerow(feature)
csv_writer.writerow(params1)
csv_writer.writerow(params2)
csv_writer.writerow(params3)
csv_writer.writerow(params4)
f.close()

result=pd.read_csv('params.csv')
kinds=list(result.index)
result=pd.concat([result,result[[feature[0]]]],axis=1)
centers=np.array(result.iloc[:,:])

n=len(feature)
angle=np.linspace(0, 2*np.pi, n, endpoint=False)
angle=np.concatenate((angle,[angle[0]]))

fig=plt.figure()
ax=fig.add_subplot(111,polar=True)

labels=['Male','Female','Married','Single']
#colors=['r','b','y','g']
print(len(kinds))
for i in range(2):
    ax.plot(angle,centers[i],linewidth=1,label=labels[i],marker='o')
    plt.fill(angle,centers[i],alpha=0.2)

ax.set_thetagrids(angle*180/np.pi, feature)
plt.title('Consumer portrait')
plt.legend(loc='best')
plt.show()
