import spacy
import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/情感和主题程序')
import xlrd
import re
import emoji
nlp = spacy.load("en_core_web_sm")

path = './各国家all/fra_result_all.txt'  #result_all_0321.txt
pathxlsx = 'Twitter_keys.xls'
pathsave = './各国家aspect/fra/fra_aspect_'
fin = open(path,'r',encoding='utf-8')
lines = fin.readlines()
fin.close()
data = xlrd.open_workbook(pathxlsx)
names = data.sheet_names()
N = 0
for i in range(0,len(data.sheets())):
    # if names[i] == 'worker':
    content_name = []
    content = []
    table = data.sheets()[i]
    for j in range(1,table.nrows): 
        datas = table.row_values(j,start_colx=0,end_colx=None)
        content.append(datas[0].lower())
    contents = list(set(content))
    for line in lines:
        content_ts = []
        if len(line.strip().split('\t')) == 3:
            sentence,time,_= line.strip().split('\t')
            sentences = nlp(sentence).sents
            for sent in sentences:
                for key in contents:
                    content_t = []
                    if len(key)!=0:
                        if key not in str(sent).split(' '):
                            continue
                        else:
                            content_t.append(names[i])
                            content_t.append(str(sent))
                            content_t.append(time)
                            content_t.append(key)
                            N+=1
                            print(N)
                        if len(content_t)!=0:
                            content_ts.append(content_t)
                    else:
                        continue
            if len(content_ts)==0:
                continue
            else:
                content_name.append(content_ts)
        else:
            continue

    pathfile = pathsave + names[i] + '.txt'
    with open(pathfile,'w',encoding='utf-8') as fin:
        for line in content_name:
            for l in line:
                for s in l:
                    fin.write(s+'#####')
                fin.write('\n')
        


