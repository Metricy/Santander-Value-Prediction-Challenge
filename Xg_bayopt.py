import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#os.chdir('C:\\Users\\Administrator\\Desktop\\kaggle_S\\input')
#data = pd.read_csv('..\\input\\train.csv')
data = pd.read_csv('../input/train.csv')
target = np.log1p(data['target'])
data.drop(['ID', 'target'], axis=1, inplace=True)

#leak = pd.read_csv('..\\input\\train_leak.csv')
leak = pd.read_csv('../input/train_leak.csv')
data['leak'] = leak['compiled_leak'].values
data['log_leak'] = np.log1p(leak['compiled_leak'].values)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5

#features_info = pd.read_csv("..\\input\\feature_report.csv")
features_info = pd.read_csv("../input/feature_report.csv")

#test = pd.read_csv('..\\input\\test.csv')
#tst_leak = pd.read_csv('..\\input\\test_leak.csv')
test = pd.read_csv('../input/test.csv')
tst_leak = pd.read_csv('../input/test_leak.csv')
test['leak'] = tst_leak['compiled_leak']
test['log_leak'] = np.log1p(tst_leak['compiled_leak'])

folds = KFold(n_splits=5, shuffle=True, random_state=1)

# Use all features for stats
features = [f for f in data if f not in ['ID', 'leak', 'log_leak', 'target']]
data.replace(0, np.nan, inplace=True)
data['log_of_mean'] = np.log1p(data[features].replace(0, np.nan).mean(axis=1))
data['mean_of_log'] = np.log1p(data[features]).replace(0, np.nan).mean(axis=1)
data['log_of_median'] = np.log1p(data[features].replace(0, np.nan).median(axis=1))
data['nb_nans'] = data[features].isnull().sum(axis=1)
data['the_sum'] = np.log1p(data[features].sum(axis=1))
data['the_std'] = data[features].std(axis=1)
data['the_kur'] = data[features].kurtosis(axis=1)

test.replace(0, np.nan, inplace=True)
test['log_of_mean'] = np.log1p(test[features].replace(0, np.nan).mean(axis=1))
test['mean_of_log'] = np.log1p(test[features]).replace(0, np.nan).mean(axis=1)
test['log_of_median'] = np.log1p(test[features].replace(0, np.nan).median(axis=1))
test['nb_nans'] = test[features].isnull().sum(axis=1)
test['the_sum'] = np.log1p(test[features].sum(axis=1))
test['the_std'] = test[features].std(axis=1)
test['the_kur'] = test[features].kurtosis(axis=1)

# Only use good features, log leak and stats for training
num_feats = 60 #could be tuned
good_features = features_info["feature"].iloc[len(features_info) - num_feats: len(features_info)]
features = good_features.tolist()
features = features + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']

from bayes_opt import bayesian_optimization

target_df = target
target = np.reshape(np.array(target_df), newshape = [-1])
train_leak_ind = np.where(data['leak'].notnull())
def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              subsample,
              colsample_bytree,
              reg_lambda,
              reg_alpha,
              silent=True,
              nthread=-1):
    oof_preds = np.zeros(data.shape[0])
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    for trn_idx, val_idx in folds.split(data):
        reg = xgb.XGBRegressor(
            booster='gbtree',
            n_estimators=int(n_estimators),
            learning_rate = learning_rate,  # 如同学习率
            min_child_weight = min_child_weight,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            max_depth=int(max_depth),  # 构建树的深度，越大越容易过拟合
            gamma = gamma,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
            #max_delta_step=0,
            subsample = subsample,  # 随机采样训练样本
            colsample_bytree= colsample_bytree,  # 生成树时进行的列采样
            reg_lambda=reg_lambda,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            reg_alpha=reg_alpha,  # L1 正则项参数
        )
        reg.fit(
            data[features].iloc[trn_idx], target[trn_idx],
            eval_set=[(data[features].iloc[val_idx], target[val_idx])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        oof_preds[val_idx] = reg.predict(data[features].iloc[val_idx], ntree_limit=reg.best_ntree_limit)
    oof_score = mean_squared_error(target[:len(data)], oof_preds[:len(data)]) ** .5
    oof_preds_leak = oof_preds
    oof_preds_leak[train_leak_ind] = np.log1p(np.array(data['leak'])[train_leak_ind])
    oof_score_leak = mean_squared_error(target[:len(data)], oof_preds_leak[:len(data)]) ** .5
    #data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(), 'leak'])
    #oof_score = mean_squared_error(target, oof_preds) ** .5
    #oof_score_leak = mean_squared_error(target, data['predictions']) ** .5
    return(-min(oof_score ,oof_score_leak))

if __name__ == "__main__":
    xgboostBO = bayesian_optimization.BayesianOptimization(xgboostcv,
                                                           {'max_depth': (3, 20),
                                                            'learning_rate': (0.01, 0.2),
                                                            'n_estimators': (50, 2000),
                                                            'gamma': (1., 0.01),
                                                            'min_child_weight': (1, 10),
                                                            'subsample': (0.5, 1),
                                                            'colsample_bytree': (0.5, 1),
                                                            'reg_lambda':(0, 3),
                                                            'reg_alpha': (0, 1)})
    xgboostBO.maximize(init_points=500, n_iter=20)
    print('-' * 53)
    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])

