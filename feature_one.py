# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:29:59 2018

@author: January
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def foo(var):
    return {
            89016252: 0,
            89016253: 1,
            89016259: 2,
            89950166: 3,
            89950167: 4,
            89950168: 5,
            90063345: 6,
            90109916: 7,
            90155946: 8,
            99104722: 9,
            99999825: 10,
            99999826: 11,
            99999827: 12,
            99999828: 13,
            99999830: 14,
    }.get(var,16)
    
def oof(var):
    return {
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
    }.get(var,16)




"""
['service_type', 'is_mix_service', 'online_time', '1_total_fee',
       '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
       'many_over_bill', 'contract_type', 'contract_time',
       'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num',
       'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
       'service1_caller_time', 'service2_caller_time', 'gender', 'age',
       'complaint_level', 'former_complaint_num', 'former_complaint_fee',
       'current_service', 'user_id']
"""
"""
c = train_data.groupby('is_mix_service').count()
"""
st1 = ''
st2 = ''
def deal_data():
    global st1
    global st2
    train_data = pd.read_csv("D:/比赛/智能套餐个性化匹配模型/data/trainData.csv")
    train_data['current_service'] = train_data['current_service'].apply(lambda x:foo(x))
    test_data = pd.read_csv("D:/比赛/智能套餐个性化匹配模型/data/test.csv")
    st1 = test_data.columns[0]
    st2 = train_data.columns[25]
    return train_data, test_data

def ToNumber(data):
    global Index
    for i in [4, 5, 20, 21]:
        data[Index[i]]=data[Index[i]].apply(pd.to_numeric, errors='ignore')
        data[Index[i]]=data[Index[i]].apply(pd.to_numeric, errors='0')
    return data
            
def creatFeature(data,x,A):
    global st1
    global st2
#    print(st1)
#    print(st2)
    train = pd.DataFrame()
    Index = data.columns
    for i in Index:
        data[i] = data[i].replace("\\N",0)
    data['2_total_fee'] = data['2_total_fee'].astype(np.float64)
    data['3_total_fee'] = data['3_total_fee'].astype(np.float64)
    fee = pd.DataFrame()
    fee['1_total_fee'] = data['1_total_fee']
    fee['2_total_fee'] = data['2_total_fee']
    fee['3_total_fee'] = data['3_total_fee']
    fee['4_total_fee'] = data['4_total_fee']

    #四个月花费平均值
    train['fee_mean'] = fee.mean(axis=1)
    #四个月花费最大值    
    train['fee_max'] = fee.max(axis=1)
    #四个月花费的极差
    train['fee_range'] = fee.max(axis=1) - fee.min(axis=1)
    #四个月花费的方差
    train['fee_std'] = fee.std(axis=1)
    #平均单次缴费金额
    train['pay_mean'] = data['1_total_fee'] / data['contract_time']
    #当前余额增长速率
    train['month_one'] = (data['1_total_fee'] - data['2_total_fee']).apply(lambda x : x*x)
    train['month_four']= (data['1_total_fee'] + data['2_total_fee']-data['3_total_fee'] - data['4_total_fee']).apply(lambda x : x*x/4)
    
    #性别的唖编码
    enc = OneHotEncoder()
    gender = np.array(data['gender']).astype(int).reshape((x,1))
    enc.fit([[0],[1],[2]])
    mate = enc.transform(gender).toarray()
    train['gender_1'] = mate[:,1]
    train['gender_2'] = mate[:,2]
    #是否固移融合套餐is_mix_service哑编码
    is_mix_service = np.array(data['is_mix_service']).reshape((x,1))
    enc.fit([[0],[1]])
    mate = enc.transform(is_mix_service).toarray()
    train['is_mix_service_0'] = mate[:,0]
    train['is_mix_service_1'] = mate[:,1]
    #套餐类型service_type哑编码
    if A:
        service_type = np.array(data[st2]).reshape((x,1))
    else:
        service_type = np.array(data[st1]).reshape((x,1))
    enc.fit([[1],[3],[4]])
    mate = enc.transform(service_type).toarray()
    train['service_type_0'] = mate[:,0]
    train['service_type_1'] = mate[:,1]
    train['service_type_2'] = mate[:,2]
    #连续超套many_over_bill哑编码
    many_over_bill = np.array(data['many_over_bill']).reshape((x,1))
    enc.fit([[1],[2]])
    mate = enc.transform(many_over_bill).toarray()
    train['many_over_bill_0'] = mate[:,0]
    train['many_over_bill_1'] = mate[:,1]
    #合约类型contract_type哑编码
    contract_type = np.array(data['contract_type']).reshape((x,1))
    enc.fit([[0],[1],[2],[3],[6],[7],[8],[9],[12]])
    mate = enc.transform(contract_type).toarray()
    train['contract_type_0'] = mate[:,0]
    train['contract_type_1'] = mate[:,1]
    train['contract_type_2'] = mate[:,2]
    train['contract_type_3'] = mate[:,3]
    train['contract_type_4'] = mate[:,4]
    train['contract_type_5'] = mate[:,5]
    train['contract_type_6'] = mate[:,6]
    train['contract_type_7'] = mate[:,7]
    train['contract_type_8'] = mate[:,8]
    #是否承诺低消用户is_promise_low_consume哑编码
    is_promise_low_consume = np.array(data['is_promise_low_consume']).reshape((x,1))
    enc.fit([[0],[1]])
    mate = enc.transform(is_promise_low_consume).toarray()
    train['is_promise_low_consume_0'] = mate[:,0]
    train['is_promise_low_consume_1'] = mate[:,1]
    #网络口径用户net_service哑编码
    net_service = np.array(data['net_service']).reshape((x,1))
    enc.fit([[0],[2],[3],[4],[9]])
    mate = enc.transform(net_service).toarray()
    train['net_service_0'] = mate[:,0]
    train['net_service_1'] = mate[:,1]

    #缴费金额处以缴费次数
    train['pay_num_times'] = data['pay_num']/data['pay_times']
    #月累计+本地数据流量
    train['last_add_local'] = data['last_month_traffic'] + data['local_trafffic_month']
    #套餐外通话总和, 通话总和
    train['service_caller'] = data['service1_caller_time'] + data['service2_caller_time']
    train['local_caller'] = train['service_caller'] + data['local_caller_time']
    #投诉重要性+1 乘上交费金历史投诉总量
    train['level_num'] = data['complaint_level'].apply(lambda x:x+1)*data['former_complaint_num']
    #x^2/(1+y) local_trafffic_month:x  local_caller_time:y
    train['lastc_month'] = data['local_trafffic_month'].apply(lambda x:x*x)/(data['local_caller_time']+1)
    #(x-y)^2/(1+y)
    train['lastc_month1'] = (data['local_trafffic_month']-data['local_caller_time']).apply(lambda x:x*x)/(data['local_caller_time']+1)
    #(x1+x2)/t  month_traffic:x1  local_trafffic_month:x2  contract_time:t
    train['lastc_month2'] = (data['month_traffic']+data['local_trafffic_month'])/(data['contract_time']+1)
    train['lastc_month3'] = (data['month_traffic']+data['local_trafffic_month'])/data['online_time']
    #(x+y)*money many_over_bill:x  is_promise_low_consume_1:y
    train['many_over'] = ((data['many_over_bill']+1) + (train['is_promise_low_consume_1']+1))*data['pay_num']
    #x*money/t   many_over_bill:x  contract_time:t
    train['many_over_time'] = (data['many_over_bill']+1)*data['pay_num']/data['contract_time']
    train['contract_time12'] = train['fee_mean']*data['contract_time']
    train['contract_time18'] = data['last_month_traffic']*data['contract_time']

    train['1_total_fee5'] = data['service1_caller_time']/data['1_total_fee']
    train['1_total_fee2'] = data['service2_caller_time']*data['3_total_fee']
    
    train['last_month_traffic3'] = data['last_month_traffic'].apply(lambda x:x*x)*data['contract_time']
    train['last_month_traffic4'] = data['contract_time'].apply(lambda x:x*x)*data['4_total_fee']
    train['last_month_traffic5'] = data['last_month_traffic'].apply(lambda x:x*x)/data['service1_caller_time']
    train['last_month_traffic7'] = data['last_month_traffic'].apply(lambda x:x*x)/data['contract_time']
    train['last_month_traffic72'] = train['last_add_local']/data['1_total_fee']

    #原始特征导入
    #train['service_type'] = data['service_type']
    #train['is_mix_service'] = data['is_mix_service']
    #train['many_over_bill'] = data['many_over_bill']
    #train['contract_type'] = data['contract_type']
    #train['is_promise_low_consume'] = data['is_promise_low_consume']
    #train['net_service'] = data['net_service']
    train['complaint_level'] = data['complaint_level']
    train['online_time'] = data['online_time']
    train['contract_time']=data['contract_time']
    train['pay_times']=data['pay_times']
    train['pay_num']=data['pay_num']
    train['last_month_traffic']=data['last_month_traffic']
    train['local_trafffic_month']=data['local_trafffic_month']
    train['local_caller_time']=data['local_caller_time']
    train['service1_caller_time']=data['service1_caller_time']
    train['service2_caller_time']=data['service2_caller_time']
    train['former_complaint_num']=data['former_complaint_num']
    train['former_complaint_fee']=data['former_complaint_fee']
    train['age'] = (data['age'].astype(np.float64)).apply(lambda x:int(x/10))
    train['1_total_fee'] = data['1_total_fee']
    train['2_total_fee'] = data['2_total_fee']
    train['3_total_fee'] = data['3_total_fee']
    train['4_total_fee'] = data['4_total_fee']

    train['2_total_fee2'] = train['2_total_fee'].apply(lambda x : x*x)
    train['2_total_fee3'] = train['3_total_fee'].apply(lambda x : x*x)
    train['2_total_fee4'] = train['4_total_fee'].apply(lambda x : x*x)
    
    train['1_total_fee8'] = data['contract_time']/data['1_total_fee']
    train['service_cha'] = data['service1_caller_time']-data['service2_caller_time']
    train['31_total_fee'] = data['online_time']/data['1_total_fee']
    train['51_total_fee'] = (data['service1_caller_time']-data['service2_caller_time'])/data['1_total_fee']

    train['age1'] = train['age'].apply(lambda x:(x*x-4*x-77)*4)
    train['age2'] = train['age1'].apply(lambda x:np.log(x))
    train['age3'] = train['age1']/data['1_total_fee']
    train['age5'] = train['1_total_fee']+train['last_month_traffic3']
    train['age6'] = train['1_total_fee']*train['1_total_fee5']
    train['age7'] = train['age5']+train['age6']

    train['user_id'] = data['user_id']
    return train


    #165491 nan
