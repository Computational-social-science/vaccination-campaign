# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:23:11 2021

@author: mikij
"""

#catboost
import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/shap/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import xgboost as xgb
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import shap
import catboost

from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from catboost import *

##load data
import pandas as pd
alf=pd.read_csv('./data/vaccination-0419.csv',encoding='utf-8')#读取数据
print(alf.shape)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
f=[column for column in alf]
feature=f[:60] #取特征数量
alf=alf[feature]
alf.head()

from sklearn.model_selection import train_test_split
f=[column for column in alf]
feature=f[:-1] #最后一列为y，取前列
x_data=alf[feature]
y_data=alf[['vaccination']] #y值
#划分训练集测试集
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=10)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

##Catboost
from catboost import CatBoostClassifier
cat_clf=CatBoostClassifier()
cat_clf.fit(train_x,train_y,eval_set=[(test_x,test_y)],)
prediction_1=cat_clf.predict_proba(test_x)# 返回预测属于某标签的概率
prediction_2=cat_clf.predict(test_x)

##shap--所有特征
shap.initjs()
shap_values_o=cat_clf.get_feature_importance(catboost.Pool(train_x,train_y),type='ShapValues')
shap_values_cat=shap_values_o[:,:-1]#二分类
#shap.summary_plot(shap_values_cat,train_x,max_display=100)

##xgboost
import xgboost as xgb
xgboost_model=xgb.XGBClassifier(scale_pos_weight=2) #learning_rate=0.02,n_estimators=200,max_depth=5,base_score=0.5
xgboost_model.fit(train_x,train_y)
prediction_1=xgboost_model.predict_proba(test_x)
prediction_2=xgboost_model.predict(test_x)

#shap
shap.initjs()
mymodel=xgboost_model.fit(train_x,train_y)
mybooster=mymodel.get_booster()

model_modify=mybooster.save_raw()[4:]
def myfun(self=None):
    return model_modify
mybooster.save_raw=myfun

explainer=shap.TreeExplainer(mybooster)
shap_values_xg=explainer.shap_values(train_x)
print(shap_values_xg.shape)
#shap.summary_plot(shap_values_xg,train_x,max_display=100)

shap_values_final=np.zeros((shap_values_xg.shape[0],shap_values_xg.shape[1]))#初始化
for i in range(shap_values_xg.shape[0]):
    for j in range(shap_values_xg.shape[1]):
        shap_values_final[i][j]=(shap_values_cat[i][j]+shap_values_xg[i][j])/2

shap.summary_plot(shap_values_final,train_x,max_display=100) #display-输出特征的个数




