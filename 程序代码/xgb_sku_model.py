import xgboost as xgb
from sklearn.cross_validation import train_test_split
import pandas as pd

if __name__=='__main__':
    print('开始训练商品模型.')
    train = pd.read_csv('./feat/train_sku_model_feat.csv')
    valid = pd.read_csv('./feat/valid_sku_model_feat.csv')
    # test = pd.read_csv('./feat/test_model_user_feat.csv')

    # X_train, X_test, y_train, y_test = train_test_split(train_x, train_x.label.values, test_size=0.2, random_state=1)
    dtrain = xgb.DMatrix(train.drop(['user_id', 'sku_id', 'brand', 'label'], axis=1), label=train.label)
    # dtest = xgb.DMatrix(test.drop(['user_id', 'sku_id', 'brand', 'label'], axis=1), label=test.label)
    dvalid = xgb.DMatrix(valid.drop(['user_id','sku_id','brand'], axis=1))
    #最终调整参数
    param = {'learning_rate' : 0.01, 'n_estimators': 1200, 'max_depth': 3,
            'min_child_weight': 5, 'gamma': 0.2, 'subsample': 0.6, 'colsample_bytree': 0.8,
            'scale_pos_weight': 1, 'eta': 0.05, 'silent': 0, 'objective': 'binary:logistic','booster': 'gbtree'}
    num_round = 600
    param['nthread'] = -1
    param['eval_metric'] = "logloss"
    evallist = [(dtest, 'test'), (dtrain, 'train')]
    bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist, early_stopping_rounds=10)

    ##预测
    predict = bst.predict(donline)
    sku_proba = valid[['user_id', 'sku_id']]
    sku_proba['sku_proba'] = predict

    ##提取每个用户购买概率最大的商品
    sku_proba = sku_proba.groupby(['user_id'], as_index=False).apply(lambda t: t[t.sku_proba == t.sku_proba.max()]).reset_index()[
        ['user_id', 'sku_id', 'sku_proba']]
    ##读取用户模型训练结果
    user_proba = pd.read_csv("./valid_user_proba.csv")

    ##按照概率值从大到小排序
    sku_proba.sort_values(by="sku_proba", ascending=False, inplace=True)
    user_proba.sort_values(by="proba", ascending=False, inplace=True)

    ##用户模型 与 商品模型 各取前600并集
    Top_user = user_proba.iloc[:600]
    Top_sku = sku_proba.iloc[:600][['user_id', 'sku_id']]
    Top_user = sku_proba[sku_proba.user_id.isin(Top_user.user_id)]
    Top_user = Top_user.groupby(['user_id'], as_index=False).apply(lambda t: t[t.sku_proba == t.sku_proba.max()]).reset_index()[
        ['user_id', 'sku_id']]

    pred = pd.concat([Top_sku, Top_user])
    pred = pred.drop_duplicates()
    pred = pred[pred.user_id.duplicated() == False]
    pred.astype(int).to_csv("valid_submit.csv", index=False)