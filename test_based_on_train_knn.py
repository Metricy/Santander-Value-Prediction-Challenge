import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./input"))

import lightgbm as lgb
from bayes_opt import BayesianOptimization

from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]

train_leak = pd.read_csv("./input/train_leak_org.csv")
test_leak = pd.read_csv("./input/test_leak_org.csv")
train_leak = train_leak.replace(np.nan,0.0)
test_leak = test_leak.replace(np.nan,0.0)

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]


# In[27]:


best_lag = 29
train_leak = rewrite_compiled_leak(train_leak, best_lag)
test_leak = rewrite_compiled_leak(test_leak, best_lag)


# In[28]:


new_train = test.loc[test_leak["compiled_leak"] > 0].copy()
new_train["target"] = test_leak["compiled_leak"].loc[test_leak["compiled_leak"] > 0]
_temp_train = train.copy()
new_train = pd.concat([_temp_train, new_train]).reset_index(drop=True)

new_train_target = np.log1p(new_train["target"])
new_train = new_train[transact_cols]

new_test = test.loc[test_leak["compiled_leak"] == 0].copy().reset_index(drop=True)
new_test = new_test[transact_cols]

total = pd.concat([new_train, new_test]).reset_index(drop=True)
total = preprocessing.scale(total)
new_train = total[:len(new_train)]
new_test = total[len(new_test):]

pca = PCA(n_components=20)
pca.fit(total)

re_train = pca.transform(new_train)

folds = KFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros(new_train.shape[0])
for train_idx, test_idx in folds.split(re_train):
    neigh = KNeighborsRegressor(n_neighbors=20, weights='distance')
    neigh.fit(re_train[train_idx], new_train_target.values[train_idx]) 
    oof_preds[test_idx] = neigh.predict(re_train[test_idx])


mean_squared_error(new_train_target[:len(train)], oof_preds[:len(train)]) ** .5
oof_preds[np.where(train_leak["compiled_leak"]>0)] = np.log1p(train_leak["compiled_leak"].loc[train_leak["compiled_leak"]>0])
mean_squared_error(new_train_target[:len(train)], oof_preds[:len(train)]) ** .5

# In[30]:







#new_test['target'] = np.expm1(new_test['target'])
#new_test.head(11)

#sub = test[["ID"]]
#sub["target"] = test_leak["compiled_leak"]
#sub.loc[sub["target"] > 0, "target"] = np.expm1(oof_preds[len(train):])
#sub.loc[sub["target"] == 0, "target"] = new_test['target'].values

# sub.to_csv(f"lgb_and_leak_{best_lag}.csv", index=False)
# print(f"lgb_and_leak_{best_lag}.csv saved")
