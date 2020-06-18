#%%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# X = np.zeros((6, 5))
# X[0] = [np.nan, 1, 2, 3, 4]
# X[1] = [1, 2, 3, 4, np.nan]
# X[2] = [np.nan, 5, 4, 3, 2]
# X[3] = [np.nan, 5, 4, 4, 1]
# X[4] = [4, np.nan, 4, 4, 1]
# X[5] = [np.nan, 2, 3, 4, 5]
# y = np.array([1, 1, 0, 0, 0, 1])

m = MLPClassifier(verbose=True, activation="relu", hidden_layer_sizes=(10, ))
data = pd.read_csv("classified5_clean.csv")

data = shuffle(data)  # 打乱
X = data.iloc[:, 2:].to_numpy()  # consumption
# discount,user_level,plus,gender,age,marital_status,education,city_level,purchase_power
y = data.iloc[:, 1].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
m.fit(X_train, y_train)
y_hat = m.predict(X_test)
print(accuracy_score(y_train, m.predict(X_train)))
print(accuracy_score(y_test, y_hat))

# %%
