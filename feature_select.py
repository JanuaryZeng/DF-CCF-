# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:17:25 2018

@author: January
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from feature_one import  creatFeature,deal_data,oof
from sklearn.decomposition import PCA

#导入数据
trainData,test_data = deal_data()
#875217 612652 262565

#处理数据
train = creatFeature(trainData,875217,True)
train_label = trainData['current_service']
test = creatFeature(test_data,200000,False)
id_test = test['user_id']
del test['user_id'],train['user_id']

#train = PCA(n_components=40).fit_transform(train)
#交叉印证集
X_train, X_test, y_train, y_test = train_test_split(train, train_label, test_size=.4, random_state=0)
#数据转换
lgb_train = lgb.Dataset(X_train,label=y_train)
lgb_eval = lgb.Dataset(X_test,label=y_test)
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True
#test = creatFeature(test_data)
params = {
            'objective': 'multiclass',
            'num_class':15,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'learning_rate': 0.3,
            'max_depth':20,
            'silent':True,
            'num_leaves':25,
            'metric': 'multi_logloss',
            #'boosting_type': 'gbdt',
            #'boosting': 'dart',
            #'max_bin':10,
            #'min_data_in_leaf':8,
            #'feature_fraction': 0.6,
            #'bagging_fraction': 1,
            #'bagging_freq':0,
            #'min_split_gain': 0
}

#自定义F1评价函数
start = datetime.now()
clf = lgb.train(params,                     # 参数字典
                lgb_train,                  # 训练集
                num_boost_round=500,        # 迭代次数
                valid_sets=lgb_eval,        # 验证集
                early_stopping_rounds=30,   # 早停系数
                feval=f1_score_vali)   
stop = datetime.now()
lgb.plot_importance(clf,figsize=(20,20))

#np.isnan(train).any()

"""
data1 = clf.predict(train)
data2 = np.argmax(data1, axis = 1)
train['predict2'] = data2

ypred1 = clf.predict(test)
ypred3 = np.argmax(ypred1,axis = 1)
test['predict2'] = ypred3
for i in tqdm(range(200000)):
    ypred3[i] = oof(ypred3[i])
sample = pd.DataFrame()
sample['user_id'] = id_test
sample['predict'] = ypred3
sample.to_csv("D:/比赛/智能套餐个性化匹配模型/to_data/submit.csv",index=None)
"""