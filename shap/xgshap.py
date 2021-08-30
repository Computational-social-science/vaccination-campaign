# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:36:28 2021

@author: mikij
"""
#xgboost
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

sns.set_style('white')#设置样式
#plt.sytle.use('seaborn')#设置样式
plt.rcParams['font.sans-serif']='Times New Roman' #字体
plt.rcParams['axes.unicode_minus']=False #字符显示

##load data
import pandas as pd
with open('./data/all-anti-vaccination-0222.csv',"rb") as f:
    alf=pd.read_csv(f)
#alf=pd.read_csv('multiTimeline.csv',encoding='utf-8')#读取数据
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
y_data=alf[['antivaccination']] #y值
#划分训练集测试集
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=10)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

#xgboost
import xgboost as xgb
xgboost_model=xgb.XGBClassifier(scale_pos_weight=2) #learning_rate=0.02,n_estimators=200,max_depth=5,base_score=0.5
xgboost_model.fit(train_x,train_y)
prediction_1=xgboost_model.predict_proba(test_x)
prediction_2=xgboost_model.predict(test_x)
#准确度--二分类
from sklearn.metrics import roc_curve,auc,roc_auc_score
false_positive_rate,true_positive_rate,thresholds=roc_curve(test_y.values,prediction_1[:,1],drop_intermediate=False)
auc_1=auc(false_positive_rate,true_positive_rate)
auc_2=roc_auc_score(test_y,prediction_2,average='macro',sample_weight=None)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
print("auc:%.2f%%,roc_auc_score:%.2f%%"%(auc_1*100,auc_2*100))

from sklearn.metrics import accuracy_score,recall_score
print("Acc:",accuracy_score(test_y,prediction_2))
print("Recall:",recall_score(test_y,prediction_2,average='micro'))

#准确度--多分类
from scipy import interp
from itertools import cycle
m=test_y.shape[0]
n=2 #类别个数
y_test=np.zeros((m,n)) #创建对应的分类标签矩阵
for i in range(m):
    j=test_y.values[i,0]
    y_test[[i],[j]]=1
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction_1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area（方法二-micro）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prediction_1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一macro）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow'])
for i, color in zip(range(n), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
#观察一下micro和macro哪种效果更好（接近左上角）
auc_1=roc_auc["micro"]
auc_2=roc_auc["macro"]
print("micro-auc:%.2f%%,macro-auc:%.2f%%"%(auc_1*100,auc_2*100))

from sklearn.metrics import accuracy_score,recall_score
print("Acc:",accuracy_score(test_y,prediction_2))
print("Recall:",recall_score(test_y,prediction_2,average='micro'))

#决策树--有问题
xgb.plot_tree(xgboost_model,num_trees=0)
fig=plt.gcf()
fig.set_size_inches(128,128)
plt.show()

#shap
shap.initjs()
mymodel=xgboost_model.fit(train_x,train_y)
mybooster=mymodel.get_booster()

model_modify=mybooster.save_raw()[4:]
def myfun(self=None):
    return model_modify
mybooster.save_raw=myfun

explainer=shap.TreeExplainer(mybooster)
shap_values=explainer.shap_values(train_x)
print(shap_values.shape)
#所有特征
shap.summary_plot(shap_values,train_x,max_display=100) #display-输出特征的个数
shap.summary_plot(shap_values,train_x,plot_type="bar",max_display=50)

#单个特征
shap.dependence_plot('clinical trial',shap_values,train_x,interaction_index=None) 
shap.dependence_plot('Vaccine trial',shap_values,train_x)

np.savetxt('44.csv',shap_values[:,77],fmt='%f',delimiter=None)

#case study
expected_value=explainer.expected_value
shap.force_plot(expected_value,shap_values[40,:],train_x.iloc[40,:])#二分类
shap.force_plot(expected_value[0],shap_values[0][40,:],train_x.iloc[40,:])#多分类

#####finetune
from numpy import sort
print(sort(xgboost_model.feature_importances_))

from sklearn.feature_selection import SelectFromModel
#print(sort(xgboost_model.feature_importances_)
thresholds=sort(xgboost_model.feature_importances_)
#选出重要的特征
for thresh in thresholds:
    selection=SelectFromModel(xgboost_model,threshold=thresh,prefit=True)
    select_X_train=selection.transform(train_x) #transform()数据标准化  
    all_name=alf.columns.values.tolist()#获得所有特征名称
    select_name_index0=selection.get_support(indices=True)#留下特征的索引值，list格式
    select_name0=[]
    for i in select_name_index0:
        select_name0.append(all_name[i])
        
    selection_model=xgb.XGBClassifier(learning_rate=0.03,n_estimators=300,max_depth=3,base_score=0.5,gamma=1,subsample=0.3,colsample_bytree=0.7,colsample_bylevel=0.4)
    #n_estimatores-迭代次数，即决策树个数 max_depth-树的深度 subsample-训练每棵树时，使用的数据占全部训练集的比例 colsample_bytree-训练每棵树时，使用的特征占全部特征的比例 colsample_bylevel-用来控制树的每一级的每一次分裂，对列数的采样的占比
    selection_model.fit(select_X_train,train_y)
    select_X_test=selection.transform(test_x)
    prediction_1=selection_model.predict_proba(select_X_test)
    prediction_2=selection_model.predict(select_X_test)
    false_positive_rate,true_positive_rate,thresholds=roc_curve(test_y.values,prediction_1[:,1])
    auc_1=auc(false_positive_rate,true_positive_rate)
    auc_2=roc_auc_score(test_y,prediction_2,average='macro',sample_weight=None)
    
    print("Thresh=%.3f,n=%d,Auc:%.2f%%,roc_auc_score:%.2f%%,f1:%.2f%%" %
          (thresh,select_X_train.shape[1],auc_1*100,auc_2*100,f1_score(test_y,prediction_2,average='micro')*100))

#准确度
from sklearn.metrics import roc_curve,auc,roc_auc_score
false_positive_rate,true_positive_rate,thresholds=roc_curve(test_y.values,prediction_1[:,1],drop_intermediate=False)
auc_1=auc(false_positive_rate,true_positive_rate)
auc_2=roc_auc_score(test_y,prediction_2,average='macro',sample_weight=None)
plt.plot(false_positive_rate,true_positive_rate)
plt.show()
print("auc:%.2f%%,roc_auc_score:%.2f%%"%(auc_1*100,auc_2*100))

from sklearn.model_selection import train_test_split
f=[column for column in alf]
feature=f[:-1] #最后一列为y，取前列
x_data=alf[select_name0]
y_data=alf[['anti-vaccination']] #y值
#划分训练集测试集
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=10)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

#xgboost
import xgboost as xgb
xgboost_model=xgb.XGBClassifier(scale_pos_weight=2) #learning_rate=0.02,n_estimators=200,max_depth=5,base_score=0.5
xgboost_model.fit(train_x,train_y)
prediction_1=xgboost_model.predict_proba(test_x)
prediction_2=xgboost_model.predict(test_x)
#准确度
from sklearn.metrics import roc_curve,auc,roc_auc_score
false_positive_rate,true_positive_rate,thresholds=roc_curve(test_y.values,prediction_1[:,1],drop_intermediate=False)
auc_1=auc(false_positive_rate,true_positive_rate)
auc_2=roc_auc_score(test_y,prediction_2,average='macro',sample_weight=None)
plt.plot(false_positive_rate,true_positive_rate)
plt.show()
print("auc:%.2f%%,roc_auc_score:%.2f%%"%(auc_1*100,auc_2*100))

#shap
shap.initjs()
mymodel=xgboost_model.fit(train_x,train_y)
mybooster=mymodel.get_booster()

model_modify=mybooster.save_raw()[4:]
def myfun(self=None):
    return model_modify
mybooster.save_raw=myfun

explainer=shap.TreeExplainer(mybooster)
shap_values=explainer.shap_values(train_x)
print(shap_values.shape)
#所有特征
shap.summary_plot(shap_values,train_x,max_display=50) #display-输出特征的个数
shap.summary_plot(shap_values,train_x,plot_type="bar",max_display=50)

t=shap.dependence_plot('coronavirus variant',shap_values,train_x,interaction_index=None,color = 'r')
t=plt.gcf()
t.set_facecolor('white')#背景颜色
ax=plt.gca()
ax.patch.set_facecolor("white")    # 设置 ax1 区域背景颜色
ax.spines['left'].set_linewidth(3) #设置边框线宽
ax.spines['right'].set_color('blue') #设置边框线颜色

#plt_light = plt.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
#plt_dark = plt.colors.ListedColormap(['g', 'r', 'b'])
#plt.figure(facecolor='w')
#plt.pcolormesh(x1, x2, y_show_hat, cmap=plt_light) #对网格点进行上色

    