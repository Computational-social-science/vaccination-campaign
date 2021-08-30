# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:25:10 2021

@author: mikij
"""
import matplotlib.colors as colors
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/shap/data')
##load data
with open('emo-vac.xlsx',"rb") as f:
    alf=pd.read_excel(f,sheet_name="in")
#alf=pd.read_csv('multiTimeline.csv',encoding='utf-8')#读取数据
print(alf.shape)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
f=[column for column in alf]
feature=f[:20] #取特征数量
alf=alf[feature]
alf.head()

#pandas自带corr相关性计算
alf.corr(method='kendall') #pearson kendall spearman
#alf.corr(method='spearman')

#画图
colorslist = ['#80E9E9','#8080FF','#Df80Df']
cmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=800)

plt.figure(figsize=(10,10))
sns_plot=sns.heatmap(alf.corr(),annot=True,cmap=cmaps)
sns_plot.tick_params(labelsize=18) # heatmap 刻度字体大小
# colorbar 刻度线设置
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15) # colorbar 刻度字体大小
plt.show()

#pmi
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt


disVar = ['worry', 'neutral', 'happiness', 'sadness', 'love','vaccination']# 需要进行归一化互信息计算的变量名（excel中的列名）
dataDis = alf.loc[:,disVar]                  # 需要归一化互信息处理的数据提取
[_, varLenDis] = dataDis.shape
nmiMatrix = [[0 for col in range(varLenDis)] for row in range(varLenDis)]
nmiMatrix = pd.DataFrame(nmiMatrix, columns=disVar, index=disVar)
for i in range(varLenDis):
    for j in range(varLenDis):
        nmiMatrix.iloc[i,j] = normalized_mutual_info_score(dataDis[disVar[i]].values, dataDis[disVar[j]].values, average_method='arithmetic')


plt.subplots(figsize=(10, 8))
plt.rcParams['font.sans-serif']=['Times New Roman']  # 正常显示文字
plt.rcParams['axes.unicode_minus']=False      # 正常显示负号
plt.rcParams.update({'font.size': 16})
sns_plot=sns.heatmap(nmiMatrix, annot=True, vmax=1, square=True, cmap=cmaps) #cmap="Blues"
sns_plot.tick_params(labelsize=20) # heatmap 刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# colorbar 刻度线设置
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=16) # colorbar 刻度字体大小
#plt.xlabel("互信息相关系数")
plt.show()



##随时间pmi
emotions = ['worry', 'neutral', 'happiness', 'sadness', 'love']

worry=alf["worry"]
neutral=alf["neutral"]
happiness=alf["happiness"]
sadness=alf["sadness"]
love=alf["love"]
vacc=alf["vaccination"]

all_list=[]
for emo in emotions:
    emo_list=[]
    print(emo)
    for i in range(16):
        num=7*(i+1)
        emo_list.append(normalized_mutual_info_score(alf[emo][:num],vacc[:num]))
    all_list.append(emo_list)

df=pd.DataFrame(all_list)
df.index=emotions
df.to_excel('C:/Users/mikij/Desktop/pmi/fra.xlsx',index=True)

