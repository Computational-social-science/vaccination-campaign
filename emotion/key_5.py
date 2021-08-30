# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:31:06 2021

@author: mikij
"""

import csv
import os
import pandas as pd
import numpy as np

path = 'C:/Users/mikij/Desktop/实验/代码/情感和主题程序/各国家aspect/0418拼接结果/fra/result_twitter_fra_aspect_country.txt_2.json.csv'

##load data
with open(path,"rb") as f:
    data=pd.read_csv(f)

nation=list(set(data['aspect']))

#统计国家出现次数
nation_dict={}
for d in data['aspect']:
    nation_dict[d] = nation_dict.get(d, 0) + 1

list1=data['aspect']
list2=data['prediction']

#统计国家对应情感出现次数
sen_dict={}
for n in nation:
    sen_list=[0,0,0,0,0,0] #['worry', 'neutral', 'happiness', 'sadness', 'love']all
    for i in range(data.shape[0]):
        if n==list1[i]:
            sen=list2[i]
            if sen=='worry':
                sen_list[0]=sen_list[0]+1
            elif sen=='neutral':
                sen_list[1]=sen_list[1]+1
            elif sen=='happiness':
                sen_list[2]=sen_list[2]+1
            elif sen=='sadness':
                sen_list[3]=sen_list[3]+1
            elif sen=='love':
                sen_list[4]=sen_list[4]+1
    sen_list[5]=sen_list[0]+sen_list[1]+sen_list[2]+sen_list[3]+sen_list[4]
    sen_dict.setdefault(n,sen_list)
            
emotions = ['worry', 'neutral', 'happiness', 'sadness', 'love','all'] #
df=pd.DataFrame(sen_dict)
df.index=emotions
df.to_excel('C:/Users/mikij/Desktop/实验/代码/情感和主题程序/各国家aspect/0418拼接结果/fra-nation.xlsx',index=True)
    
        
