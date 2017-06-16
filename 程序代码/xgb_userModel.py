import xgboost as xgb
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

if __name__=='__main__':
    print('开始训练用户模型.')
    train = pd.read_csv('./feat/train_user_model_feat.csv')
    valid = pd.read_csv('./feat/valid_user_model_feat.csv')
    # test = pd.read_csv('./feat/test_user_feat.csv')

    # X_train, X_test, y_train, y_test = train_test_split(train_x, train_x.label.values, test_size=0.2, random_state=1)
    dtrain = xgb.DMatrix(train.drop(['user_id', 'label'], axis=1), label=train.label)
    # dtest = xgb.DMatrix(test.drop(['user_id', 'label'], axis=1), label=test.label)
    dvalid = xgb.DMatrix(valid.drop(['user_id', 'label'], axis=1))
    #最终调整参数
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3,
            'min_child_weight': 5, 'gamma': 0.0, 'subsample': 1, 'colsample_bytree': 0.8,
            'scale_pos_weight': 1, 'eta': 0.05, 'silent': 0, 'objective': 'binary:logistic','booster': 'gbtree'}
    num_round = 400
    param['nthread'] = -1
    param['eval_metric'] = "logloss"
    evallist = [(dtest, 'test'), (dtrain, 'train')]
    bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist, early_stopping_rounds=10)

    #预测用户购买第8类商品的概率
    predict = bst.predict(dvalid)
    dvalid_proba = dvalid[['user_id']]
    dvalid_proba['proba'] = predict
    dvalid_proba.to_csv("./valib_proba_user.csv", index=False)
    print("用户模型训练完成.")