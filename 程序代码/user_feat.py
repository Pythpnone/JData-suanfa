# coding: utf-8
import pandas as pd
import numpy as np
import datetime as dt
import math
import os
import re
import pickle

ACTION_DATA = 'JData_Action.csv'
USER_FILE = 'JData_User.csv'
PRODUCT_FILE = 'JData_Product.csv'
COMMENT_FILE = 'JData_Comment.csv'


#11号到16号购买过第8类商品的用户作为测试集
def Action_useful_user():
    dump_path = './cache/Action_useful_data'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = pd.read_csv(ACTION_DATA)[['user_id','sku_id','time','cate','type']]
        #4-11号到16号买了第八类商品,并且16号前7天有第8类商品浏览记录的用户保存下来
        test_sample= dataset[(dataset['time']>'2016-04-11 00:00:00')&(dataset['type']==4)&(dataset['cate']==8)][['user_id','sku_id']].drop_duplicates()
        #11号前7天浏览过第八类商品的用户
        sample_data = dataset[(dataset['time']<'2016-04-11 00:00:00')&(dataset['time']>'2016-04-04 00:00:00')&(dataset['cate']==8)].user_id.to_frame().drop_duplicates()
        test_sample = pd.merge(sample_data,test_sample,on='user_id',how='left')
        test_sample = test_sample[test_sample.sku_id.notnull()]
        test_sample.to_csv('train_label_dataset.csv',index=False)
        #删除11号之前购买过第八类商品的用户
        dataset = dataset[dataset['time']<'2016-04-11 00:00:00']
        delta_user_id = pd.DataFrame(dataset[(dataset['type']==4)&(dataset['cate']==8)]['user_id']).drop_duplicates()
        delta_user_id['label'] = 0
        all_user_id = pd.merge(dataset,delta_user_id,on='user_id',how='left')
        all_user_id = pd.DataFrame(all_user_id[all_user_id.label.isnull()]['user_id']).drop_duplicates()

#         all_user_id = dataset.drop(delta_user_id.user_id.isin(dataset.user_id),axis=0)
#         all_user_id = pd.DataFrame(all_user_id['user_id']).drop_duplicates()
        df = pd.merge(all_user_id,dataset,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#选择在预测期间前7天与第8类用户有交互的用户（购买过第8类商品的去除）来构造样本
def sample_area_user_id():
    dump_path = './cache/Action_user_id_sample_data'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(ACTION_DATA)[['user_id','sku_id','type','time','cate']]
        df1 = df
        df1 = pd.DataFrame(df[(df['cate']==8)&(df['time']>'2016-04-09 00:00:00')])
        #购买过第8类商品的用户
        delta_data = df1[df1['type']==4].user_id.to_frame().drop_duplicates()
        delta_data['label'] = 0
        df2 = pd.merge(df1,delta_data,on='user_id',how='left')
        df2 = df2[df2.label.isnull()].user_id.to_frame().drop_duplicates()
        #df2为预测前7天没买过第八类商品但是看过第八类商品的用户
        df = pd.merge(df2,df,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df

#训练集01-31 到04-11号 ,样本区间04-04 00:00:00--04-11 00:00:00,标签区间2016-04-11 00:00:00 --2016-04-16 00:00:00
#保存04号到11号查看了第八类商品的用户，作为样本用户
def create_train_set():
    dump_path = './cache/train_dataset'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df =pickle.load(f)
    else:
        #去除只在04号到11号登陆，并且购买了商品的用户
        dataset = Action_useful_user()
        df1 = dataset[dataset['time']<'2016-04-11 00:00:00']
        # 04-04 --04-11
        df3 = pd.DataFrame(df1[(df1['time']>'2016-04-04 00:00:00')&(df1['cate']==8)]['user_id']).drop_duplicates()
        df = pd.merge(df3,df1,on='user_id',how='left')
        #删除非第8类商品浏览记录
        df = df[df['cate']==8]
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#用户属性特征
def feat_setup_1():
    dump_path = './cache/feat_setup_1'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        df = create_train_set()
        user_info = pd.read_csv(USER_FILE,encoding='gbk')
        #查看用户完整数据
        df = pd.merge(df,user_info,on='user_id',how='left')
        #用户等级大于2的
        df = df[df['user_lv_cd']>=2]
        #注册日期距离测试第一天的距离（单位：天）
        df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'],format='%Y-%m-%d')
        df['test_time'] = '2016-04-11'
        df['test_time'] = pd.to_datetime(df['test_time'],format='%Y-%m-%d')
        df['reg_dis'] = (df['test_time'] - df['user_reg_tm']).apply(lambda t:t.total_seconds()/60/60/24)
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


# # 用户时间行为特征
def feat_setup_2():
    dump_path = './cache/feat_setup_2'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_1()
        df = dataset[['user_id','time']]
        #登录天数
        df['time'] = df.time.apply(lambda t:t[:10])
        df = df.drop_duplicates()
        type_dummies = pd.get_dummies(df['time'],prefix=None)
        type_dummies.columns = ['71','70','69','68','67','66','65','64','63','62','61','60','59','58',\
            '57','56','55','54','53','52','51','50','49','48','47','46','45','44','43','42','41','40','39','38','37','36','35',\
            '34','33','32','31','30','29','28','27','26','25','24','23','22','21','20','19','18','17','16','15','14','13','12',\
            '11','10','9','8','7','6','5','4','3','2','1']
        df1 = pd.concat([df,type_dummies],axis=1)
        df1 = df1.drop(['time'],axis=1)
        df1 = df1.drop_duplicates()
        df1 = df1.groupby('user_id',as_index=False).sum()
        df1['load_days'] = df1.sum(axis=1) - df1['user_id']
        df = pd.merge(dataset,df1,on='user_id',how='left')
        
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#第一次登录距离预测第一天的距离(天)
def feat_setup_3():
    dump_path = './cache/feat_setup_3'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_2()
        df = dataset[['user_id','time']]
        df= df.groupby('user_id',as_index=False).time.min()
        df.columns = ['user_id','first_time']
        df['first_time'] = df.first_time.apply(lambda t:t[:10])
#         df = df.drop(['time'],axis=1)
        df = df.drop_duplicates()
        df['first_time'] = pd.to_datetime(df['first_time'],format='%Y-%m-%d')
        df['pre_time'] = '2016-04-11'
        df['pre_time'] = pd.to_datetime(df['pre_time'],format='%Y-%m-%d')
        df['first_lo_dis'] = (df['pre_time'] - df['first_time']).apply(lambda t:t.total_seconds()/60/60/24)
        df = df.drop(['pre_time','first_time'],axis=1)
        df = pd.merge(dataset,df,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#最后一次浏览第8类商品时间与预测第一天距离（秒）
def feat_setup_4(end_time = '2016-04-11 00:00:00'):
    dump_path = './cache/feat_setup_4'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_3()
        df = dataset[['user_id','time']]
        df = df.groupby('user_id',as_index=False).time.max()
        df['time'] = pd.to_datetime(df.time)
        df['last_tm_dis'] = df.time.apply(lambda t:(pd.to_datetime(end_time) - t).total_seconds())
        df = df.drop(['time'],axis=1)
        df = df.drop_duplicates()
        df = pd.merge(dataset,df,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#用户前（1/2/3/5/7）天的有效行为时间(单位：秒)
def feat_setup_5():
    dump_path = './cache/feat_setup_5'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_4(end_time = '2016-04-11 00:00:00')
        df = dataset[['user_id','time']]
        #前一天有效行为时间
        df1 = df[df['time']>'2016-04-10 00:00:00']
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_1_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_1_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        
        #前两天有效时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-09 00:00:00')&(df['time']<'2016-04-10 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_2_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_2_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_2_day'] = dataset['belong_2_day'] + dataset['belong_1_day']
        
        # 前3天有效时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-08 00:00:00')&(df['time']<'2016-04-09 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_3_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_3_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_3_day'] = dataset['belong_3_day'] + dataset['belong_2_day']
        
        #前4天有效行为时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-07 00:00:00')&(df['time']<'2016-004-08 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_4_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_4_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_4_day'] = dataset['belong_4_day'] + dataset['belong_3_day']
        
        #前5天有效行为时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-06 00:00:00')&(df['time']<'2016-04-07 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_5_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_5_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_5_day'] = dataset['belong_5_day'] + dataset['belong_4_day']
        
        #前6天有效行为时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-05 00:00:00')&(df['time']<'2016-04-06 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_6_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_6_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_6_day'] = dataset['belong_6_day'] + dataset['belong_5_day']
        
        #前7天有效行为时间
        df = dataset[['user_id','time']]
        df1 = df[(df['time']>'2016-04-04 00:00:00')&(df['time']<'2016-04-05 00:00:00')]
        df2 = df1.groupby('user_id',as_index=False).time.min()
        df2['time'] = pd.to_datetime(df2.time,format='%Y-%m-%d')
        df3 = df.groupby('user_id',as_index=False).time.max()
        df3.columns = ['user_id','max_time']
        df3['max_time'] = pd.to_datetime(df3.max_time,format='%Y-%m-%d')
        df3 = pd.merge(df2,df3,on='user_id',how='left')
        df3['belong_7_day'] = (df3['max_time'] - df3['time']).apply(lambda t:t.total_seconds())
        df3 = df3[['user_id','belong_7_day']].drop_duplicates()
        dataset = pd.merge(dataset,df3,on='user_id',how='left')
        dataset['belong_7_day'] = dataset['belong_7_day'] + dataset['belong_6_day']
        
        # df = dataset[['user_id','belong_1_day','belong_2_day','belong_3_day','belong_4_day','belong_5_day','belong_6_day','belong_7_day']].drop_duplicates()
        df = dataset
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
        
    return df


# # 用户行为特征
#用户前（7/15/60天）第八类商品的操作数
def feat_setup_6():
    dump_path = './cache/feat_setup_6'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_5()
        #用户前7天对第八类商品的操作数
        df = dataset[dataset['time']>'2016-04-04 00:00:00'][['user_id','type']]
        df = df.groupby('user_id',as_index=False).count()
        df.columns = ['user_id','last_7']
        #用户前15天对第八类商品的操作数
        df1 = dataset[dataset['time']>'2016-03-27 00:00:00'][['user_id','type']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','last_15']
        #用户前60天对第八类商品的操作数
        df2 = dataset[dataset['time']>'2016-02-11 00:00:00'][['user_id','type']]
        df2 = df2.groupby('user_id',as_index=False).count()
        df2.columns = ['user_id','last_60']
        dataset = pd.merge(dataset,df,on='user_id',how='left')
        dataset = pd.merge(dataset,df1,on='user_id',how='left')
        df = pd.merge(dataset,df2,on='user_id',how='left')
        df = df.fillna(0)
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#用户前7/15/60天的所有操作数
def feat_setup_7():
    dump_path = './cache/feat_setup_7'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = Action_useful_user()
        df = dataset[dataset['time']>'2016-04-04 00:00:00'][['user_id','type']]
        #用户前7天总操作数
        df = df.groupby('user_id',as_index = False).count()
        df.columns = ['user_id','last_7_all']
        #用户前15天总操作数
        df1 = dataset[dataset['time']>'2016-03-27 00:00:00'][['user_id','type']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','last_15_all']
        #用户总操作数
        df2 = dataset[dataset['time']>'2016-02-11 00:00:00'][['user_id','type']]
        df2 = df2.groupby('user_id',as_index=False).count()
        df2.columns = ['user_id','last_60_all']
        dataset = pd.DataFrame(dataset['user_id']).drop_duplicates()
        dataset = pd.merge(dataset,df,on='user_id',how='left')
        dataset = pd.merge(dataset,df1,on='user_id',how='left')
        dataset = pd.merge(dataset,df2,on='user_id',how='left')
        
        df = feat_setup_6()
        df = pd.merge(df,dataset,on='user_id',how='left')
        df = df.fillna(0)
        df['ratio_7'] = df['last_7'] / df['last_7_all']
        df['ratio_15'] = df['last_15'] / df['last_15_all']
        df['ratio_60'] = df['last_60'] / df['last_60_all']
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#用户前7天加购物车/删除购物车/关注统计
def feat_setup_8():
    dump_path ='./cache/feat_setup_8'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_7()
        df = dataset[dataset['time']>'2016-04-04 00:00:00'][['user_id','type']]
        df_2 = df[df['type']==2]
        df_3 = df[df['type']==3]
        df_4 = df[df['type']==5]
        df_2 = df_2.groupby('user_id',as_index=False).count()
        df_2.columns = ['user_id','cart_num']
        df_3 = df_3.groupby('user_id',as_index=False).count()
        df_3.columns = ['user_id','delte_num']
        df_4 = df_4.groupby('user_id',as_index=False).count()
        df_4.columns = ['user_id','like_num']
        dataset = pd.merge(dataset,df_2,on='user_id',how='left')
        dataset = pd.merge(dataset,df_3,on='user_id',how='left')
        df = pd.merge(dataset,df_4,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#预测前7/15天浏览第8类商品天数
def feat_setup_9():
    dump_path = './cache/feat_setup_9'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_8()
        #前七天登陆天数（查看第8类商品）
        df = dataset[['user_id','7','6','5','4','3','2','1']].drop_duplicates()
        df['cate_8_days/7'] = df.sum(axis=1) - df['user_id']
        df = df[['user_id','cate_8_days/7']]
        #前15天........
        df1 = dataset[['user_id','15','14','13','12','11','10','9','8','7','6','5','4','3','2','1']].drop_duplicates()
        df1['cate_8_days/15'] = df1.sum(axis=1) - df1['user_id']
        df1 = df1[['user_id','cate_8_days/15']]
        dataset = pd.merge(dataset,df,on='user_id',how='left')
        df = pd.merge(dataset,df1,on='user_id',how='left')
        
        #前7天/15天总的登陆天数
        data = Action_useful_user()[['user_id','time']]
        data = data[(data['time']>'2016-03-27 00:00:00')&(data['time']<'2016-04-11 00:00:00')]
        data['time'] = data.time.apply(lambda t:t[:10])
        data = data.drop_duplicates()
        type_dummies = pd.get_dummies(data['time'],prefix=None)
        type_dummies.columns = ['15','14','13','12','11','10','9','8','7','6','5','4','3','2','1']
        data = pd.concat([data,type_dummies],axis=1)
        data = data.drop(['time'],axis=1)
        data = data.groupby('user_id',as_index=False).sum()

        #前15天登录天数
        data['15_days_load'] = data.sum(axis=1) - data['user_id']
        #前7天登录天数
        data = data[['user_id','7','6','5','4','3','2','1','15_days_load']]
        data['7_days_load'] = data.sum(axis=1) - data['user_id'] - data['15_days_load']
        data = data[['user_id','7_days_load','15_days_load']]
        
        #两张表拼接
        df = pd.merge(df,data,on='user_id',how='left')
        df['ratio_cate_8/7'] = df['cate_8_days/7'] / df['7_days_load']
        df['ratio_cate_8/15'] = df['cate_8_days/15'] / df['15_days_load']
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#统计前7天0点到4点，4点到8点，八点到10点，12点到16点，16点到20点，20点到24点6个时区中用户浏览总量
def feat_setup_10():
    dump_path = './cache/feat_setup_10'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_9()
        df = dataset[['user_id','time','type']]
        df = df[df['time']>'2016-04-04 00:00:00']
        df['time'] = df.time.apply(lambda t:int(t[11:13]))
        #0点到4点
        df1 = df[df['time']<4][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','0-4_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        #4点到8点
        df1 = df[(df['time']>=4)&(df['time']<8)][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','4-8_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        #8点到12点
        df1 = df[(df['time']>=8)&(df['time']<12)][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','8-12_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        #12点到16点
        df1 = df[(df['time']>=12)&(df['time']<16)][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','12-16_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        #16点到20点
        df1 = df[(df['time']>=16)&(df['time']<20)][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','16-20_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        #20点到24点
        df1 = df[(df['time']>=20)&(df['time']<24)][['user_id','time']]
        df1 = df1.groupby('user_id',as_index=False).count()
        df1.columns = ['user_id','20-24_bro_num']
        df = pd.merge(df,df1,on='user_id',how='left')
        
        df = pd.merge(dataset,df,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


#用户仅仅在预测前三天登录过，并且登陆天数==1
def feat_rule_1():
    dump_path = './cache/feat_rule_1'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_setup_10()
        df = dataset[dataset['time']>'2016-04-08 00:00:00']
        df = df[df['load_days']==1]
        df['label_1'] = 1
        df = df[['user_id','label_1']].drop_duplicates()
        dataset = pd.merge(dataset,df,on='user_id',how='left')
        #预测前三天加入过购物车、删除购物车、关注的用户
        df = dataset[dataset['time']>'2016-04-08 00:00:00']
        df = df[df['cate']==8]
        df = df[(df['type']==2)|(df['type']==3)|(df['type']==5)]
        df['label_2'] = 1
        df = df[['user_id','label_2']].drop_duplicates()
        dataset = pd.merge(dataset,df,on='user_id',how='left')
        
        #年龄特征
        df= pd.DataFrame(dataset[dataset['age']=='-1']['user_id']).drop_duplicates()
        df['age_feat'] = 0.1002
#         dataset = pd.merge(dataset,df,on='user_id',how='left')
        
        df1= pd.DataFrame(dataset[dataset['age']=='16-25岁']['user_id']).drop_duplicates()
        df1['age_feat'] = 0.1236
#         dataset = pd.merge(dataset,df,on='user_id',how='left')
        
        df2= pd.DataFrame(dataset[dataset['age']=='26-35岁']['user_id']).drop_duplicates()
        df2['age_feat'] = 0.1281
#         dataset = pd.merge(dataset,df,on='user_id',how='left')
        
        df3= pd.DataFrame(dataset[dataset['age']=='36-45岁']['user_id']).drop_duplicates()
        df3['age_feat'] = 0.1285
#         dataset = pd.merge(dataset,df,on='user_id',how='left')
        
        df4= pd.DataFrame(dataset[dataset['age']=='46-55岁']['user_id']).drop_duplicates()
        df4['age_feat'] = 0.1
        df = pd.concat([df,df1,df2,df3,df4],axis=0)
        df = pd.merge(dataset,df,on='user_id',how='left')
        
        #前3天0-4,20-24点加入购物车的用户
        df = dataset[['user_id','time','type']]
        df = df[df['time']>'2016-04-08 00:00:00']
        df['time'] = df.time.apply(lambda t:int(t[11:13]))
        #0点到4点加入购物车
        df2 = df[df['time']<4][['user_id','type']]
        df2 = pd.DataFrame(df2[(df2['type']==2)|(df2['type']==5)]['user_id']).drop_duplicates()
        df2['0-4_cart_or_care'] = 1
        dataset = pd.merge(dataset,df2,on='user_id',how='left')
        #20点到24点
        df1 = df[df['time']>=20][['user_id','type']]
        df1 = pd.DataFrame(df1[(df1['type']==2)|(df1['type']==5)]['user_id']).drop_duplicates()
        df1['20-24_cart_or_care'] = 1
        df = pd.merge(dataset,df1,on='user_id',how='left')
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df


# # 用训练集，测试集构建模型
#提取关键特征，构建模型
def user_train_model():
    dump_path = './cache/user_train_model_data'
    if os.path.exists(dump_path):
        with open(dump_path,'rb') as f:
            df = pickle.load(f)
    else:
        dataset = feat_rule_1()
        dataset = dataset.drop(['sku_id','time','type','71','70','69','68','67','66','65','64','63','62','61','60','59','58',\
            '57','56','55','54','53','52','51','50','49','48','47','46','45','44','43','42','41','40','39','38','37','36','35',\
            '34','33','32','31','30','29','28','27','26','25','24','23','22','21','20','19','18','17','16','15','14','13','12','11','10','9','8','7','6','5','4','3','2','1'],axis=1)
        df = dataset.drop(['cate','user_reg_tm','test_time','age'],axis=1)
        # df.to_csv('./feat/train_user_feat.csv',index=False)
        with open(dump_path,'wb') as f:
            pickle.dump(df,f)
    return df

def get_user_model():
    df = Action_useful_user()
    df = sample_area_user_id()
    df = create_train_set()
    df = feat_rule_1()
    df = feat_rule_2()
    df = feat_rule_3()
    df = feat_rule_4()
    df = feat_rule_5()
    df = feat_rule_6()
    df = feat_rule_7()
    df = feat_rule_8()
    df = feat_rule_9()
    df = feat_rule_10()
    df = user_train_model()
    label_dataset = pd.read_csv('train_label_dataset.csv')[['user_id']].drop_duplicates()
    label_dataset['label'] = 1
    df = pd.merge(df,label_dataset,on='user_id',how='left').drop_duplicates()
    df.to_csv('./feat/train_user_model_feat.csv',index=False)
    return df

