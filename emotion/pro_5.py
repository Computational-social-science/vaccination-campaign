import csv
import os
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
path = 'C:/Users/mikij/Desktop/实验/代码/情感和主题程序/各国家aspect/0418拼接结果/fra/'
files = os.listdir(path)
catogory = []
total = 0 
emotions = ['worry', 'neutral', 'happiness', 'sadness', 'love'] #
value_all = []

for file in files:
    # emotions = []
    values = []
    path1 = path + file
    aspect = file.split('.')[0].split('_')[-1]
    catogory.append(aspect)
    data_t = pd.read_csv(path1,encoding='utf-8',error_bad_lines=False)
    data = data_t.drop_duplicates() #去重
    sen_list=[0,0,0,0,0]
    for sen in data['prediction']:
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
    
    sentiments = data['prediction'].value_counts() 
    total = sum(sentiments) #- sentiments['neutral'] - sentiments['worry']  
    values.append(sen_list[0]/total)
    values.append(sen_list[1]/total)
    values.append(sen_list[2]/total)
    values.append(sen_list[3]/total)
    values.append(sen_list[4]/total)
    '''
    values.append(sentiments['worry']/total)
    values.append(sentiments['neutral']/total)
    values.append(sentiments['happiness']/total)
    values.append(sentiments['sadness']/total)
    values.append(sentiments['love']/total)
    '''
    value_all.append(values)

#存表
df=pd.DataFrame(value_all, columns=emotions)
df.index=catogory
df.to_excel('C:/Users/mikij/Desktop/实验/代码/情感和主题程序/各国家aspect/0418拼接结果/fra_re.xlsx',index=True)


fig = plt.figure()
for values in value_all:
    angles=np.linspace(0, 2*np.pi,len(values), endpoint=False)
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)     
    ax.set_thetagrids(angles * 180/np.pi, emotions)
    ax.set_ylim(0,1)
ax.grid(True)
plt.title(aspect)
plt.show()

right = np.array(positive)
left = np.array(negative)
plt.xticks([])
plt.yticks(range(len(right)),catogory)
rect_a = plt.barh(range(len(right)), right, color='#ED7C30', height=0.5)
rect_b = plt.barh(range(len(left)), -left, color='#9DC3E7', height=0.5)
plt.legend((rect_b,rect_a),('negative','positive'),bbox_to_anchor=(0.5, 1.1), frameon=False,loc='upper center', ncol=2)
plt.show()