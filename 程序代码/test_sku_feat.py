# coding: utf-8
import pandas as pd
import numpy as np
import re
import datetime as dt
import pickle
import math
import os

ACTION_DATA = 'JData_Action.csv'
USER_FILE = 'JData_User.csv'
PRODUCT_FILE = 'JData_Product.csv'
COMMENT_FILE = 'JData_Comment.csv'


# # 商品时间行为特征
# 第一次浏览该商品到最后一次浏览该商品的时差，用户对该商品的关注度,精确到秒
def sku_feat_setup_1():
    dump_path = './cache/sku_feat_setup_1_test'
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        with open('./cache/feat_rule_1_test', 'rb') as f:
            dataset = pickle.load(f)[['user_id', 'sku_id', 'time', 'type', 'cate_8_days/7', 'cate_8_days/15']]
        df = dataset[['user_id', 'sku_id', 'time']]
        df1 = df
        df1 = df1.groupby(['user_id', 'sku_id'], as_index=False).time.min().drop_duplicates()
        df1.columns = ['user_id', 'sku_id', 'point_min_tm']
        df = df.groupby(['user_id', 'sku_id'], as_index=False).time.max().drop_duplicates()
        df.columns = ['user_id', 'sku_id', 'point_max_tm']
        df = pd.merge(df, df1, on=('user_id', 'sku_id'), how='left').drop_duplicates()
        df['point_min_tm'] = pd.to_datetime(df.point_min_tm, format='%Y-%m-%d')
        df['point_max_tm'] = pd.to_datetime(df.point_max_tm, format='%Y-%m-%d')
        df['attend_tm'] = (df['point_max_tm'] - df['point_min_tm']).apply(
            lambda t: t.total_seconds() / 60 / 60 / 24)  # 用户关注该商品天数
        df1 = df[df['attend_tm'] == 0]
        df1['attend_tm'] = 1 / (60 * 60 & 24)
        df2 = df[df['attend_tm'] != 0]
        df = pd.concat([df1, df2], axis=0)
        df = pd.merge(dataset, df, on=('user_id', 'sku_id'), how='left')
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df


# 对于商品最后一次加购/删除时间
def sku_feat_setup_2(start_date='2016-01-30 00:00:00', end_date='2016-04-06 00:00:00'):
    dump_path = './cache/sku_feat_setup_2_test'
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        dataset = sku_feat_setup_1()
        df = dataset[['user_id', 'sku_id', 'time', 'type']]
        # 最后一次加购时间
        last_cart_tm = df[df['type'] == 2].drop_duplicates()
        last_cart_tm = last_cart_tm.groupby(['user_id', 'sku_id', 'type'], as_index=False).time.max()
        last_cart_tm = last_cart_tm.drop(['type'], axis=1)
        last_cart_tm.columns = ['user_id', 'sku_id', 'last_cart_tm']
        # 最后一次删除时间
        last_del_tm = df[df['type'] == 3].drop_duplicates()
        last_del_tm = last_del_tm.groupby(['user_id', 'sku_id', 'type'], as_index=False).time.max()
        last_del_tm = last_del_tm.drop(['type'], axis=1)
        last_del_tm.columns = ['user_id', 'sku_id', 'last_del_tm']
        # 比较加购时间和删除时间
        df = pd.merge(last_cart_tm, last_del_tm, on=('user_id', 'sku_id'), how='left')
        df = df.fillna(start_date)
        df1 = df[df['last_cart_tm'] > df['last_del_tm']]
        df1 = df1.drop(['last_del_tm'], axis=1)
        df1['big_ratio_sku'] = 1
        # 加购时间小于删除时间
        df2 = df[df['last_cart_tm'] <= df['last_del_tm']]
        df2['low_ratio_sku'] = -1
        df2 = df2.drop(['last_cart_tm'], axis=1)
        df = pd.merge(df1, df2, on=('user_id', 'sku_id'), how='left')

        # 最后一次加购时间距离测试时间（单位/秒）
        temp_df = df[df['big_ratio_sku'] == 1][['user_id', 'sku_id', 'last_cart_tm']].drop_duplicates()
        temp_df['last_cart_tm'] = pd.to_datetime(temp_df.last_cart_tm, format='%Y-%m-%d')
        temp_df['cart_att_tm'] = temp_df.last_cart_tm.apply(lambda t: (pd.to_datetime(end_date) - t).total_seconds())
        temp_df = temp_df.drop(['last_cart_tm'], axis=1)
        df = pd.merge(df, temp_df, on=('user_id', 'sku_id'), how='left')
        df = pd.merge(dataset, df, on=('user_id', 'sku_id'), how='left')
        df = df.drop(['last_cart_tm', 'last_del_tm'], axis=1)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df


# 用户对该商品的点击次数和平均点击次数（越高说明用户对该商品关注度越大）
def sku_feat_setup_3():
    dump_path = './cache/sku_feat_setup_3_test'
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        dataset = sku_feat_setup_2(start_date='2016-01-30 00:00:00', end_date='2016-04-06 00:00:00')
        df = dataset[['user_id', 'sku_id', 'attend_tm', 'type']]
        df = df.groupby(['user_id', 'sku_id', 'attend_tm'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'attend_tm', 'point_num']
        # 平均点击次数
        df['mean_point'] = df['point_num'] / df['attend_tm']

        # 最后七天对该商品的点击数和平均点击数（点击总数/最后七天查看第八类商品的天数）
        temp_df = dataset[['user_id', 'sku_id', 'type', 'cate_8_days/7', 'time']]
        temp_df = temp_df[temp_df['time'] > '2016-03-30 00:00:00'][['user_id', 'sku_id', 'type', 'cate_8_days/7']]
        temp_df = temp_df.groupby(['user_id', 'sku_id', 'cate_8_days/7'], as_index=False).count()
        temp_df.columns = ['user_id', 'sku_id', 'cate_8_days/7', 'last_7_point']

        # 最后7天点击次数最大的加个标签
        temp = temp_df[['user_id', 'sku_id', 'last_7_point']]
        temp = temp.groupby(['user_id', 'sku_id'], as_index=False).last_7_point.max()
        temp['max_pt_label'] = 1
        temp_df = pd.merge(temp_df, temp, on=('user_id', 'sku_id', 'last_7_point'), how='left')
        # 最后七天平均点击数
        temp_df['mean_7_pt_num'] = temp_df['last_7_point'] / temp_df['cate_8_days/7']
        df = pd.merge(df, temp_df, on=('user_id', 'sku_id'), how='left')
        df = df.drop(['attend_tm', 'cate_8_days/7'], axis=1).drop_duplicates()
        df = pd.merge(dataset, df, on=('user_id', 'sku_id'), how='left')
        df = df.drop(['time', 'type', 'point_max_tm', 'point_min_tm'], axis=1).drop_duplicates()
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df


# 拼接用户浏览商品表和商品信息表，商品评论表
def all_sku_info():
    dump_path = './cache/all_sku_info_test'
    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = sku_feat_setup_3()
        product = pd.read_csv(PRODUCT_FILE)
        comment = pd.read_csv(COMMENT_FILE)
        df = pd.merge(df, product, on='sku_id', how='left')
        df = pd.merge(df, comment, on='sku_id', how='left')
        df = df.drop(['dt', 'cate'], axis=1)
        df = df[df['has_bad_comment'] < 0.1].drop_duplicates()
        # 保留评论数最高的那一行
        temp_df = df[['user_id', 'sku_id', 'comment_num']]
        temp_df = temp_df.groupby(['user_id', 'sku_id'], as_index=False).comment_num.max()
        temp_df.columns = ['user_id', 'sku_id', 'comment_num']
        df = df.drop(['comment_num'], axis=1).drop_duplicates()
        df = pd.merge(df, temp_df, on=('user_id', 'sku_id'), how='left').drop_duplicates()
        df = df.fillna(0)
        with open(dump_path, 'wb') as f:
            pickle.dump(df, f)
    return df


def get_sku_model():
    df = sku_feat_setup_1()
    df = sku_feat_setup_2()
    df = sku_feat_setup_3()
    df = all_sku_info()
    label_dataset = pd.read_csv('test_label_dataset.csv')
    label_dataset['label'] = 1
    df = pd.merge(df, label_dataset, on=('user_id', 'sku_id'), how='left')
    df.to_csv('./feat/train_sku_model_feat.csv', index=False)
    return df
