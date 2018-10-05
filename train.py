# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:29:59 2018

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

#导入数据
trainData,test_data = deal_data()

#处理数据
train = creatFeature(trainData,262565,True)
train_label = trainData['current_service']
test = creatFeature(test_data,200000,False)
id_test = test['user_id']
del test['user_id'],train['user_id']

params = {
            #'boosting_type': 'gbdt',
            #'boosting': 'dart',
            'objective': 'multiclass',
            #'metric': 'multi_logloss',
            'num_class':15,

            'learning_rate': 0.1,
            #'num_leaves':25,
            'max_depth':4,

            #'max_bin':10,
            #'min_data_in_leaf':8,
            'silent':True,

            #'feature_fraction': 0.6,
            #'bagging_fraction': 1,
            #'bagging_freq':0,

            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            #'min_split_gain': 0
}

#自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

start = datetime.now()

xx_score = []
cv_pred = []
n_splits = 5
seed = 33

X = train.values
y = train_label
X_test = test.values

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
for index,(train_index,test_index) in enumerate(skf.split(X,y)):
    print(index)

    X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf=lgb.train(params,train_data,num_boost_round=350,valid_sets=[validation_data],early_stopping_rounds=50,feval=f1_score_vali,verbose_eval=1)

    xx_pred = clf.predict(X_valid,num_iteration=clf.best_iteration)

    xx_pred = [np.argmax(x) for x in xx_pred]

    xx_score.append(f1_score(y_valid,xx_pred,average='weighted'))

    y_test = clf.predict(X_test,num_iteration=clf.best_iteration)

    y_test = [np.argmax(x) for x in y_test]

    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

stop = datetime.now()

submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
label2current_service = {
           0: 89016252,
           1: 89016253,
           2: 89016259,
           3: 89950166,
           4: 89950167,
           5: 89950168,
           6: 90063345,
           7: 90109916,
           8: 90155946,
           9: 99104722,
           10: 99999825,
           11: 99999826,
           12: 99999827,
           13: 99999828,
           14: 99999830,
    }

df_test = pd.DataFrame()
df_test['id'] = list(id_test.unique())
df_test['predict'] = submit
df_test['predict'] = df_test['predict'].map(label2current_service)

df_test.to_csv('D:/比赛/智能套餐个性化匹配模型/to_data/submit.csv',index=False)
#保存模型
joblib.dump(clf,'D:/比赛/智能套餐个性化匹配模型/模型/9_19_one.pkl')

