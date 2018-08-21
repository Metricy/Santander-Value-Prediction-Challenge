import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("./input"))

import lightgbm as lgb
from catboost import Pool, CatBoostRegressor
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

train_leak = pd.read_csv("./input/train_leak_081415_111sets.csv")
test_leak = pd.read_csv("./input/test_leak_081415_7854_111sets.csv")
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


best_lag = 37
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

# new_test.replace(0, np.nan, inplace=True)
# new_test['log_of_mean'] = np.log1p(new_test[features].replace(0, np.nan).mean(axis=1))
# new_test['mean_of_log'] = np.log1p(new_test[features]).replace(0, np.nan).mean(axis=1)
# new_test['log_of_median'] = np.log1p(new_test[features].replace(0, np.nan).median(axis=1))
# new_test['nb_nans'] = new_test[features].isnull().sum(axis=1)
# new_test['the_sum'] = np.log1p(new_test[features].sum(axis=1))
# new_test['the_std'] = new_test[features].std(axis=1)
# new_test['the_kur'] = new_test[features].kurtosis(axis=1)

# In[17]:


def cat_evaluate(iterations, learning_rate, depth,
                reg_lambda, bagging_temperature, od_wait,
                feature_criterion):
    
    good_features = report.loc[report['rmse'] <= feature_criterion]["feature"].values
    features = good_features.tolist()
    features = features + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']
    cat_train = new_train[features].copy()
    oof_preds = np.zeros(new_train.shape[0])
    
    for trn_idx, val_idx in folds.split(new_train):
        train_pool = Pool(cat_train.iloc[trn_idx], target.iloc[trn_idx])
        valid_pool = Pool(cat_train.iloc[val_idx], target.iloc[val_idx])
    
        model = CatBoostRegressor(iterations=int(iterations),
                              learning_rate=learning_rate,
                              depth=int(depth),
                              reg_lambda = reg_lambda,
                              bootstrap_type = "Bayesian",
                              bagging_temperature = bagging_temperature,
                              od_type='Iter',
                              od_wait=int(od_wait),
                              random_seed = 3,
                              eval_metric='RMSE')
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
        oof_preds[val_idx] = model.predict(cat_train.iloc[val_idx])

    #oof_preds[new_train['leak'].notnull()] = np.log1p(new_train.loc[new_train['leak'].notnull(), 'leak'])
    result = mean_squared_error(target[:len(train)], oof_preds[:len(train)]) ** .5
    return -result


# In[23]:

#new_test['target'] = 0

num_iter = 200
init_points = 20

catBO = BayesianOptimization(cat_evaluate, {'iterations': (50, 500),
                                            'learning_rate': (0.01,0.5),
                                            'depth': (1, 13),
                                            'reg_lambda': (0.01, 170),
                                            'bagging_temperature': (1, 100),
                                            'od_wait': (1, 60),
                                            'feature_criterion': (0.7906, 0.7997),
                                           })

catBO.initialize({"target": [-0.5673582106138704, -0.54164, -0.5272591078913655],
               'iterations': [107.99900841514871, 106, 183.5],
               'learning_rate': [0.19691072178786717, 0.3089, 0.3108079208513621],
               'depth': [6.090326126064177, 2.9349, 4.505457210624931],
               'reg_lambda': [43.836381835277145, 32.3049, 19.976033639025626],
               'bagging_temperature': [6.140075362260194, 2.2641, 8.772214238048157],
               'od_wait': [6.82669559532599, 23.9047, 9.65768728481989],
               'feature_criterion': [0.7906639802351747, 0.7962, 0.7947769162735652],
              })


print("---------------catBO.maximize---------------")
catBO.maximize(init_points=init_points, n_iter=num_iter)
print(catBO.res['max'])



#new_test['target'] = np.expm1(new_test['target'])
#new_test.head(11)

#sub = test[["ID"]]
#sub["target"] = test_leak["compiled_leak"]
#sub.loc[sub["target"] > 0, "target"] = np.expm1(oof_preds[len(train):])
#sub.loc[sub["target"] == 0, "target"] = new_test['target'].values

# sub.to_csv(f"lgb_and_leak_{best_lag}.csv", index=False)
# print(f"lgb_and_leak_{best_lag}.csv saved")
