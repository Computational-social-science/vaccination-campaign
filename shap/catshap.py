# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:36:16 2021

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

sns.set_style('white')#设置样式
#plt.sytle.use('seaborn')#设置样式
plt.rcParams['font.sans-serif']='Times New Roman' #字体
plt.rcParams['axes.unicode_minus']=False #字符显示

##load data
import pandas as pd
alf=pd.read_csv('./data/all-anti-vaccination-0222.csv',encoding='utf-8')#读取数据
print(alf.shape)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
f=[column for column in alf]
feature=f[:60] #取特征数量
alf=alf[feature]
alf.head()

##preprocessing
fs=[column for column in alf]
k=[]
for f in fs:
    for index,row in alf.iterrows():
        if isinstance(row[f],str) and "——" in row[f]:#替换字符
            alf[f]=alf[f].astype(str).replace('——',' ').astype(float)
        elif isinstance(row[f],str) and ">" in row[f]:
            if(f not in k):
                k.append(f)
            alf.loc[index,f]=float(row[f].replace('>',''))
        elif isinstance(row[f],str) and "<" in row[f]:
            alf.loc[index,f]=float(row[f].replace('<',''))
            if(f not in k):
                k.append(f)
alf[k]=alf[k].astype(float)
alf=alf.select_dtypes(include=['int','float'])
alf.head()

from sklearn.model_selection import train_test_split
f=[column for column in alf]
feature=f[:-1] #最后一列为y，取前列
x_data=alf[feature]
y_data=alf[['anti-vaccination']] #y值
#划分训练集测试集
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=10)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

##Catboost
from catboost import CatBoostClassifier
cat_clf=CatBoostClassifier()
cat_clf.fit(train_x,train_y,eval_set=[(test_x,test_y)],)
prediction_1=cat_clf.predict_proba(test_x)# 返回预测属于某标签的概率
prediction_2=cat_clf.predict(test_x)
#准确度--二分类
from sklearn.metrics import roc_curve,auc,roc_auc_score
false_positive_rate,true_positive_rate,thresholds=roc_curve(test_y.values,prediction_1[:,1],drop_intermediate=False)
auc_1=auc(false_positive_rate,true_positive_rate)
auc_2=roc_auc_score(test_y,prediction_2,average='macro',sample_weight=None)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0, 1], [0, 1], 'k--')
plt.show() #ROC曲线越接近左上角，该分类器的性能越好
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


##shap--所有特征
shap.initjs()
shap_values_o=cat_clf.get_feature_importance(catboost.Pool(train_x,train_y),type='ShapValues')
shap_values=shap_values_o[:,:-1]#二分类
shap_values0=shap_values_o[:,:,:-1]#多分类
shap_values=shap_change(shap_values0) #三维数组转为list

shap.summary_plot(shap_values,train_x,max_display=100) #display-输出特征的个数 这个图只支持2分类

shap.summary_plot(shap_values,train_x,plot_type="bar",max_display=150) #可以多分类

#单个特征以及2个特征的协同关系
shap.dependence_plot('religious belief',shap_values,train_x,interaction_index=None)#多分类时shap_values[0]
shap.dependence_plot('Vaccine trial',shap_values,train_x)

##shap_values[:,0]对应输入的第一个关键词
#np.savetxt('44.csv',shap_values[:,119],fmt='%f',delimiter=None)
#np.savetxt('45.csv',train_x['vaccination controversy-topic'],fmt='%f',delimiter=None)


#case study--jupyter
expected_value=shap_values_o[0,-1] #二分类
#多分类
expected_value=[]
for i in range(n):
    expected_value.append(shap_values_o[0,i,-1])

t=shap.force_plot(expected_value,shap_values[40,:],train_x.iloc[40,:])
shap.force_plot(expected_value,shap_values[40,:],train_x.iloc[40,:])#二分类
shap.force_plot(expected_value[0],shap_values[0][40,:],train_x.iloc[40,:])#多分类

t.set_facecolor('white') # 把背景颜色设置为白色
t.savefig('filename.png', bbox_inches = 'tight', dpi=500) # 保存图片


#决策树--还有问题
pool=catboost.Pool(
        data=train_x,
        label=train_y
        )
cat_clf.plot_tree(
        tree_idx=1
        )

def shap_change(shap_values):
    [m,n,f]=shap_values.shape
    new_list=[]
    for i in range(n):
        new_np=np.zeros((m,f))
        for j in range(m):
            for p in range(f):
                new_np[j,p]=shap_values[j,i,p]
            #print(j)
        new_list.append(new_np)
    return new_list
    

    