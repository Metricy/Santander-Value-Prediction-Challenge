import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./input"))

import lightgbm as lgb
from bayes_opt import BayesianOptimization

from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

print("---------------Data Loading---------------")
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]

train_leak = pd.read_csv("./input/train_leak_081613_114sets.csv")
test_leak = pd.read_csv("./input/test_leak_081613_7854_114sets.csv")
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


best_lag = 36
train_leak = rewrite_compiled_leak(train_leak, best_lag)
test_leak = rewrite_compiled_leak(test_leak, best_lag)


# In[28]:


new_train = test.loc[test_leak["compiled_leak"] > 0].copy()
new_train["target"] = test_leak["compiled_leak"].loc[test_leak["compiled_leak"] > 0]
new_train["leak"] = new_train["target"]
new_train['log_leak'] = np.log1p(new_train["leak"])

_temp_train = train.copy()
_temp_train["leak"] = train_leak['compiled_leak']
_temp_train['log_leak'] = np.log1p(_temp_train["leak"])

new_train = pd.concat([_temp_train, new_train]).reset_index(drop=True)
#new_test = test.loc[test_leak["compiled_leak"] == 0].copy().reset_index(drop=True)
#new_test['leak'] = 0
#new_test['log_leak'] = 0


# In[29]:


report = pd.read_csv("./input/feature_report.csv")

#rmses = report.loc[report['rmse'] <= 0.7925, 'rmse'].values


# In[30]:

print("---------------Features Constructing---------------")
target = np.log1p(new_train['target'])

folds = KFold(n_splits=5, shuffle=True, random_state=1)

features = [f for f in new_train if f not in ['ID', 'leak', 'log_leak', 'target']]

new_train.replace(0, np.nan, inplace=True)
new_train['log_of_mean'] = np.log1p(new_train[features].replace(0, np.nan).mean(axis=1))
new_train['mean_of_log'] = np.log1p(new_train[features]).replace(0, np.nan).mean(axis=1)
new_train['log_of_median'] = np.log1p(new_train[features][features].replace(0, np.nan).median(axis=1))
new_train['nb_nans'] = new_train[features].isnull().sum(axis=1)
new_train['the_sum'] = np.log1p(new_train[features].sum(axis=1))
new_train['the_std'] = new_train[features].std(axis=1)
new_train['the_kur'] = new_train[features].kurtosis(axis=1)

new_train["weight"] = 1

# new_test.replace(0, np.nan, inplace=True)
# new_test['log_of_mean'] = np.log1p(new_test[features].replace(0, np.nan).mean(axis=1))
# new_test['mean_of_log'] = np.log1p(new_test[features]).replace(0, np.nan).mean(axis=1)
# new_test['log_of_median'] = np.log1p(new_test[features].replace(0, np.nan).median(axis=1))
# new_test['nb_nans'] = new_test[features].isnull().sum(axis=1)
# new_test['the_sum'] = np.log1p(new_test[features].sum(axis=1))
# new_test['the_std'] = new_test[features].std(axis=1)
# new_test['the_kur'] = new_test[features].kurtosis(axis=1)

# In[17]:


def lgb_evaluate(num_leaves, subsample, colsample_bytree,
                min_split_gain, reg_alpha, reg_lambda,
                min_child_weight, learning_rate, feature_criterion,
                no_leak_weight):
    
    good_features = report.loc[report['rmse'] <= feature_criterion]["feature"].values
    features = good_features.tolist()
    features = features + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']

    new_train["weight"].loc[np.isnan(new_train['leak'])] = no_leak_weight

    dtrain = lgb.Dataset(data=new_train[["weight"] + features], 
                     label=target, free_raw_data=False)
    dtrain.construct()
    oof_preds = np.zeros(new_train.shape[0])
    
    for trn_idx, val_idx in folds.split(new_train):
            
        lgb_params = {
            'objective': 'regression',
            'num_leaves': int(num_leaves),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_split_gain': min_split_gain,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'verbose': -1,
            'seed': 3,
            'boosting_type': 'gbdt',
            'max_depth': -1,
            'learning_rate': learning_rate,
            'metric': 'l2',
            'weight_column': 0,
        }
        
        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=10000, 
            early_stopping_rounds=100,
            verbose_eval=0
        )
        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
    #oof_preds[new_train['leak'].notnull()] = np.log1p(new_train.loc[new_train['leak'].notnull(), 'leak'])
    result = mean_squared_error(target[np.isnan(new_train['leak'])], oof_preds[np.isnan(new_train['leak'])]) ** .5
    return -result


# In[23]:

#new_test['target'] = 0

num_iter = 100
init_points = 50

lgbBO = BayesianOptimization(lgb_evaluate, {'num_leaves': (8, 300),
                                            'subsample': (0.1,1),
                                            'colsample_bytree': (0.1, 1),
                                            'min_split_gain': (0.000005, 6),
                                            'reg_alpha': (0, 30),
                                            'reg_lambda': (0, 90),
                                            'min_child_weight': (0,8),
                                            'learning_rate': (0.01, 0.5),
                                            'feature_criterion': (0.7906, 0.7997),
                                            'no_leak_weight': (1,15),
                                           })

# best lag 36
lgbBO.initialize({"target": [-1.382524, -1.38085, -1.38028],
               'num_leaves': [12.02, 109.5153, 10.0000],
               'subsample': [0.1, 0.10, 0.1000],
               'colsample_bytree': [0.9999998944771263, 1.0000, 1.0000],
               'min_split_gain': [1e-05, 1e-05, 1e-05],
               'reg_alpha': [2.926647409600368, 10.0000, 10.0000],
               'reg_lambda': [0.27039842736589326, 60.0000, 40.7422],
               'min_child_weight': [0.0, 2, 2.0000],
               'learning_rate': [0.1, 0.5, 0.5000],
               'feature_criterion': [0.7907, 0.7906, 0.7997],
               'no_leak_weight': [1.1, 10, 10.0000]
              })


print("---------------lgbBO.maximize---------------")
lgbBO.maximize(init_points=init_points, n_iter=num_iter)
print(lgbBO.res['max'])



#new_test['target'] = np.expm1(new_test['target'])
#new_test.head(11)

#sub = test[["ID"]]
#sub["target"] = test_leak["compiled_leak"]
#sub.loc[sub["target"] > 0, "target"] = np.expm1(oof_preds[len(train):])
#sub.loc[sub["target"] == 0, "target"] = new_test['target'].values

# sub.to_csv(f"lgb_and_leak_{best_lag}.csv", index=False)
# print(f"lgb_and_leak_{best_lag}.csv saved")
