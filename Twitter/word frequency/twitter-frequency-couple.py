# import nltk
# nltk.download()
#from  string import maketrans
import datetime
import numpy
import numpy as np
import xlrd, xlwt
import openpyxl
import pandas as pd
import collections
import time
import os
import re
import operator
import math
import operator
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#f=open('F-C.csv','w')
def cipin(s,diction):
    s_dict = {}
    for item in s:
        if item not in diction:
            if item not in s_dict:
                s_dict[item]=1
            else:
                s_dict[item]+=1

    return s_dict
def clear1(comment,diction):
    sent = []

    results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*', re.S)
    comments = results.sub("", comment)
    remove = '‘"@#%*' + "'’“”"
    table = str.maketrans('_', '_', remove)
    comments = comments.translate(table)
    fenju = sent_tokenize(comments)
    for i in fenju:
        cixing = pos_tag(word_tokenize(i), 'universal')
        sent.append(cixing)
    # sent.append(sent_tokenize(comments))
    pos_table = ['NOUN', 'VERB', 'ADJ', 'ADV', 'NUM']

    words = []
    total=[]
    for i in sent:
        for j in i:
            if (j[1] in pos_table) and (j[1] not in diction):
                words.append(j[0])
        for k in range(0,len(words)):
            try:
                a=words[k]+' '+words[k+1]
                total.append(a)
            except IndexError:
                pass

    return total
if __name__ == '__main__':
    stop_list=[]
    with open("stop.txt", "r",encoding='utf-8-sig') as f:
        for stop in f.readlines():
            stop_list.append(stop.strip('\n').strip())# 去掉列表中每一个元素的换行符
    #print(stop_list)
    all=[]
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    i = 1
    head = ['word','fre']  # 标题
    for j in range(0, len(head)):  # 写标题
        sheet1.write(0, j, head[j])
    #path1='C:/Users/49838/Desktop/推特数据汇总-487'
    path1='C:/Users/49838/Desktop/model'
    dirs1 = os.listdir(path1)
    for file in dirs1:
        print(file)
        intput_file = file
        path=path1 + '/' + intput_file
        wb = xlrd.open_workbook(path)
        sheets = wb.sheet_names()
       # data_xls = pd.ExcelFile(path)
       # sheet_name_list=data_xls.sheet_names
        for name in sheets:

            data = pd.read_excel(path,sheet_name=name)#,sheet_name=sheet_names[i]

            list1 = data.values.tolist()
            #list1 = np.array(list1)
            for sentence in list1:
                wor=clear1(sentence[0], stop_list)
                all=wor+all
    c=dict(collections.Counter(all))            #all.append(wor)
    #sent=[]

    # print(len(all))

   # print(cipin_dic)
    cipin_dic = sorted(c.items(), key=operator.itemgetter(1), reverse=True)
    #print(cipin_dic)
    print('hh')
    j=1
    for zidian in cipin_dic:
            sheet1.write(j, 0, zidian[0])
            sheet1.write(j, 1, zidian[1])
            j=j+1
            if j%500==0:
                f.save('twitter-fre.xlsx')
            if j>1000:
                break

        #all=all+list1[0]

