import os
os.chdir('C:/Users/mikij/Desktop/实验/代码/情感和主题程序')
import xlrd
import re
import emoji

def sentence_clean(sentence):
    sentence = re.sub(r'[^\x00-\x7F]+','', sentence)
    sentence = re.sub(' rt ','', sentence)
    sentence = re.sub('(\.)+','.', sentence)
    sentence = re.sub('((www\.[^\s]+))','',sentence)
    sentence = re.sub('((http://[^\s]+))','',sentence)
    sentence = re.sub('((https://[^\s]+))','',sentence)
    sentence = re.sub('@[^\s]+','',sentence)
    sentence = re.sub('[\s]+', ' ', sentence)
    sentence = re.sub(r'#([^\s]+)', r'\1', sentence)
    sentence = re.sub('\$','',sentence)
    #sentence = re.sub('%','',sentence)
    sentence = re.sub('^','',sentence)
    sentence = re.sub('&','',sentence)
    sentence = re.sub('\*','',sentence)
    sentence = re.sub('\(','',sentence)
    sentence = re.sub('\)','',sentence)
    sentence = re.sub('-','',sentence)
    sentence = re.sub('\+','',sentence)
    sentence = re.sub('=','',sentence)
    sentence = re.sub('"','',sentence)
    sentence = re.sub('\t','',sentence)
    sentence = re.sub('\r','',sentence)
    sentence = re.sub('\n','',sentence)
    sentence = re.sub('~','',sentence)
    sentence = re.sub('`','',sentence)
    sentence = re.sub('^-?[0-9]+$','', sentence)
    #sentence = sentence.strip('\'"')
    return sentence
#location = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
#location=['UK','Bath','Birmingham','Bradford','Brighton & Hove','Bristol','Cambridge','Canterbury','Carlisle','Chester','Chichester','Coventry','Derby','Durham','Ely','Exeter','Gloucester','Hereford','Kingston upon Hull','Lancaster','Leeds','Leicester','Lichfield','Lincoln','Liverpool','London','Manchester','Newcastle upon Tyne','Norwich','Nottingham','Oxford','Peterborough','Plymouth','Portsmouth','Preston','Ripon','Salford','Salisbury','Sheffield','Southampton','St Albans','Stoke-on-Trent','Sunderland','Truro''Wakefield','Wells','Westminster','Winchester','Wolverhampton','Worcester','York','Armagh','Belfast','Londonderry','Lisburn','Newry','Aberdeen','Dundee','Edinburgh','Glasgow','Inverness','Stirling','Perth','Bangor','Cardiff','Newport','St David','Swansea']
#location=['Canada','Ottawa','Fredericton','Halifax','Cape Breton','Toronto','Hamilton','Kitchener','St. Catharines','Windsor','Oshawa','Barrie','Kingston','Guelph','Sudbury','Thunder Bay','Québec','Montréal','Sherbrooke','Trois-Rivières','Chicoutimi','Winnipeg','Victoria','Vancouver','Abbotsford','Kelowna','Charlottetown','Edmonton','Calgary','Regina','Saskatoon','Saint-John','Yellowknife','Whitehorse','Iqaluit']
#location=['Australia','Canberra','New South Wales ','Sydney','Albury','Armidale','Bathurst','Broken Hill','Cessnock','Coffs Harbour','Dubbo','Gosford','Goulburn','Grafton','Griffith','Lake Macquarie','Lismore','Maitland','Newcastle','Nowra','Orange','Port Macquarie','Queanbeyan','Tamworth','Tweed Heads','Wagga Wagga','Wollongong','Wyong','Victoria','Melbourne','Ararat','Benalla','Ballarat','Bendigo','Geelong','Mildura','Shepparton','Swan Hill','Wangaratta','Warrnambool','Wodonga','Toowoomba','Townsville','Perth','Albany','Broome','Bunbury','Geraldton','Fremantle','Kalgoorlie','Mandurah','Port Hedland','Adelaide','Mount Gambier','Merivale Bridge','Port Augusta','Port Pirie','Port Lincoln','Victor Harbor','Whyalla','Tasmania','Hobart','Burnie','Devonport','Maryborough','Queensland','Brisbane','Bundaberg','Cairns','Gladstone','Gold Coast','Gympie','Hervey Bay','Ipswich','Logan City','Mackay','Mt. Isa','Rockhampton','Sunshine Coast','Port Arthur','Launceston','Ross','Richmond','Bruny Island','St Helens','Cradle Mountain','Gordon Dam','Northern Territory ','Darwin','Alice Springs','Katherine','Wilton']
location=['France','Melun','Versailles','Nanterre','Créteil','Châlons-en-Champagne','Charleville-Mézières','Reims','Chaumont','Laon','Rouen','Orléans','Chartres','Tours','Caen','Alençon','Nevers','Auxerre','Arras','Nancy','Épinal','Colmar','Lons-le-Saunier','Belfort','Laval','LaRoche-sur-Yon','Rennes','Quimper','Poitiers','LaRochelle','Bordeaux','Mont-de-Marsan','Pau','Foix','Auch','Tarbes','Montauban','Tulle','Bourg-en-Bresse','Privas','Grenoble','Chambéry','Clerment-Ferrand','Moulins','LePuy','Montpellier','Nîmes','Perpignan','Digne','Toulon','Ajaccio','Cayenne','Fort-de-France','Saint-Denis','Angers','Niort','Lyon','Clerment-Ferrand','Nice','Paris','Evry','Bobigny','Pontoise','Troyes','Amiens','Beauvais','Évreux','Bourges','Châteauroux','Blois','Saint-Lô','Dijon','Macon','Lille','Metz','Bar-le-Duc','Strasbourg','Besançon','Vesoul','Nantes','LeMans','Saint-Brieuc','Vannes','Angoulême','Périgueux','Agen','Toulouse','Rodez','Cahors','Albi','Limoges','Guéret','Valence','Saint-Étienne','Annecy','Aurillac','Carcassonne','Mende','Marseille','Gap','Avignon','Bastia','Basse-Terre']
#location=['India','Agra','Ahmedabad','Allahabab','Bangalore','Baroda','Benares/Varanasi','Bombay','Calcutta','Coimbatore','Guntur','Hyderabad','Indore','Jaipur','Kanpru','Lucknow','Madras','Madurai','Nagpur','New Delhi','Poona','Simla','Trivandrom','Vellore','Vishakapatnam','Andhra Pradesh','Assam','Bihar''Goa','Gujarat','Haryana','Himachal Pradesh','Kanatak','Kerala','Madhya Pradesh','Maharashtra','Manpur','Meghalaya','Mizoram','Nagaland','Orissa','Punjab','Rajasthan','Tamil Nadu','Tripura','Utar Pradesh','West Bengal']


location2id = {}
id2location = {}
for idx,ls in enumerate(location):
    l = ls.lower()
    location2id[l] = idx
    id2location[idx] = l
path = 'E:/BaiduNetdiskDownload/推特36'
pathres = './fra_result_all.txt'
files = os.listdir(path)
contents = []
for file in files:
    print(file)
    data = xlrd.open_workbook(path+'\\'+file)
    for i in range(0,len(data.sheets())):
        table = data.sheets()[i]
        for j in range(1,table.nrows): #
            #nrows = table.nrows
            content = []
            datas = table.row_values(j,start_colx=0,end_colx=None)
            #print(datas)
            if datas[2]=="":
                continue
            else:
                loc = re.split('[,/]',datas[2]) 
                for lo in loc:
                    lo = lo.lower()
                    if location2id.get(lo) is None:
                        continue
                    else:
                        sentence = datas[0]
                        sentence = emoji.demojize(sentence)
                        sentence_t = sentence_clean(sentence)
                        content.append(sentence_t)
                        content.append(datas[1])
                        content.append(lo)
                        contents.append(content)
                        break

    print(len(contents))
with open(pathres,'w',encoding='utf-8') as fin:
    for line in contents:
        for l in line:
            fin.write(l+'\t')
        fin.write('\n')



# timeline = data.groupby(['time', 'prediction']).agg('count').reset_index() #**{'tweets': ('id', 'count')} #'location' 'count'
# timeline1 = data.groupby(['time']).agg('count').reset_index()
# j = 0
# n = 0
# for i in range(len(timeline)-1):
#     if timeline.ix[i,0]==timeline.ix[i+1,0]:
#         n += 1
#         timeline.ix[i,2] = timeline.ix[i,2]/timeline1.ix[j//5,1]
#         j += 1
#     else:
#         if n == 4:
#             n = 0
#             timeline.ix[i,2] = timeline.ix[i,2]/timeline1.ix[j//5,1]
#             j += 1
#         else:
#             timeline.ix[i,2] = timeline.ix[i,2]/timeline1.ix[j//5,1]
#             j += 5-n 
#             n = 0
#     if i == len(timeline)-2:
#         timeline.ix[i+1,2] = timeline.ix[i+1,2]/timeline1.ix[j//5,1]