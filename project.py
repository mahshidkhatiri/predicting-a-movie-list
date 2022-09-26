import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row, SparkSession
import csv
from pyspark.ml.recommendation import ALS 
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col
from pyspark.sql.types import *
import math
import random
def dataToCSV(file,k,header,file_name):
    list_rate=[]
    with open('ml-100k/'+file) as fp:
        line = fp.readline()
        while(line):
            z=line.rstrip("\n")
            x=z.split(k)
            list_rate.append(x)
            line=fp.readline()
    with open(file_name+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(list_rate)
def makingfileready():
    dataToCSV('u.data', "\t",['UserID','MovieID','Rating','Timestamp'],'ratings')
    dataToCSV("u.item", "|",["MovieID" ,"movie title" , "release date" , "video release date" ,
                  "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
                  "Children's" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" ,
                  "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
                  "Thriller" , "War" , "Western" ],"movies")
    dataToCSV("u.user","|", ['UserId','Age',"gender","occupation","zip code"],"users")
    with open('provider_item.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(["provider","MovieID"])
    genre=["unknown" , "Action" , "Adventure" , "Animation" ,
               "Children's" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" ,
                "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
                "Thriller" , "War" , "Western" ]
    Movies= pd.read_csv('movies.csv')
    provider_size=[]
    for i in genre:
        l=0
        for j in range(0,len(Movies.index)):
            if Movies[i][j]==1:
                l+=1
        provider_size.append([i,l])
        
    with open('provider_size.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["provider","provider_size"])
        writer.writerows(provider_size)
    item_provider=[]
    for i in range(0,len(Movies.index)):
        l=""
        for j in genre:
            if Movies[j][i]==1:
                l=l+str(genre.index(j))+","
        l=l[:-1]
        item_provider.append([i,l])
    with open('item_provider.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["item","provider"])
        writer.writerows(item_provider)
def five_foldCrossValid():
    df = pd.read_csv('ratings.csv')
    file = open("ratings.csv")
    csvreader = csv.reader(file)
    header = next(csvreader)
    listOfDFRows = df.to_numpy().tolist()
    listOfDFRows.sort(key = lambda i: i[0])
    train_5fold=[[],[],[],[],[]]
    test_5fold=[[],[],[],[],[]]
    users=[]
    l=0
    for j in range(1,6041):
        user=[]
        for i in range(l,len(listOfDFRows)):
            if j==listOfDFRows[i][0]:
                user.append(listOfDFRows[i])
            else:
                l=i
                break;
        users.append(user)
    for k in users:
        m=len(k)
        j=int(m/5)
        a=0
        b=j-1
        for i in range(0,5):
            test=[]
            train=[]
            for o in k[0:a]:
                train.append(o)
            for o in k[b+1:]:
                train.append(o)
            for o in k[a:b+1]:
                test.append(o)
            if len(k)-b<2*j:
                a+=j
                j=len(k)-b
                b+=j
            else:
                a=a+j
                b=b+j
            train_5fold[i].extend(train)
            test_5fold[i].extend(test)
    for j in range(0,5):
        s="train"+ str(j)+".csv"
        s1="test"+ str(j)+".csv"
        with open(s, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user", "item","rating","timestamp"])
            for i in train_5fold[j]:
                writer.writerow(i)
        with open(s1, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["user", "item","rating","timestamp"])
            for i in test_5fold[j]:
                writer.writerow(i)
def ratingMatrix(q):
    spark = SparkSession.builder.getOrCreate()
    train=spark.read.option("header",True).csv("train"+str(q)+".csv")
    df = train.withColumn("user",train.user.cast(IntegerType())).withColumn("item",train.item.cast(IntegerType())).withColumn("rating",train.rating.cast(DoubleType()))
    list_forper=[]
    i= pd.read_csv('movies.csv')
    u = pd.read_csv('users.csv')
    index = u.index
    index2 = i.index
    for i in range(1,len(index)+1):
        for j in range(1,len(index2)+1):
            list_forper.append([i,j,0.0])
    pandasDF = pd.DataFrame(list_forper, columns = ["user", "item","perdiction"]) 
    print(pandasDF)
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    sparkDF=spark.createDataFrame(pandasDF) 
    sparkDF.printSchema()
    sparkDF.show()
    df3 = sparkDF.withColumn("user",sparkDF.user.cast(IntegerType())).withColumn("item",sparkDF.item.cast(IntegerType()))
    df3.printSchema()
    
    als = ALS(rank=10, maxIter=10,  userCol='user', itemCol='item', seed=0 ,ratingCol='rating')
    model = als.fit(df)
    predictions =model.transform(df3.select(["user", "item"]))
    predictionPandas=predictions.toPandas().sort_values(['user', 'item'])
    predictionPandas = predictionPandas.reset_index(drop=True)
    
    list_rate=[]
    l=[]
    for i in range(0,len(predictionPandas.index)):
        if i%len(index2)==0:
            l=[predictionPandas["prediction"][i]]
        else:
            l.append(predictionPandas["prediction"][i])
            if i%len(index2)==len(index2)-1:
                list_rate.append(l)
    with open('preference_score'+str(q)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list_rate)
def reRank(q):
    w_score = pd.read_csv('preference_score'+str(q)+'.csv' , header=None)
    score = np.array(w_score.values)
    
    sorted_score = []
    for i in range(len(score)):
        sorted_score.append(np.argsort(-score[i]))
    orginal_rcomendation_list=[]
    for i in sorted_score:
        orginal_rcomendation_list.append(i[:100])
    orginal_rcomendation_list = [[v+1 for v in r] for r in orginal_rcomendation_list]
    with open('orginal_rcomendation_list'+str(q)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(orginal_rcomendation_list)
    item_number = 1682
    user_number = 943
    provider_number = 19
    m = user_number
    n = item_number
    provider_num = provider_number
    p_size= pd.read_csv('provider_size.csv')
    p_size = np.array(p_size.values)
    item_provider= pd.read_csv('item_provider.csv')
    item_provider = np.array(item_provider.values)
    i_p=[]
    for i in range(len(item_provider)):
        l=item_provider[i][1].split(",")
        l = [ int(x) for x in l ]
        i_p.append(l)
    
    k = 10
    
    total_exposure = 0
    for i in range(k):
        total_exposure += 1 / math.log((i + 2), 2)
    total_exposure = total_exposure * m
    
    fair_exposure = []
    for i in range(provider_num):
        fair_exposure.append(total_exposure / n * p_size[i][1])
    ideal_score = [0 for i in range(m)]
    for user_temp in range(m):
        for rank_temp in range(k):
            item_temp = sorted_score[user_temp][rank_temp]
            ideal_score[user_temp] += score[user_temp][item_temp] / math.log((rank_temp + 2), 2)
    
    user_satisfaction = [0 for i in range(m)]
    user_satisfaction_total = 0
    provider_exposure_score = [0 for i in range(provider_num)]
    rec_result = np.full((m, k), -1)
    rec_flag = []
    for i in range(m):
        rec_flag.append(list(sorted_score[i]))
    
    for top_k in range(k):
        # sort user according to user_satisfaction
        rank_user_satisfaction = [temp for temp in range(m)]
        random.shuffle(rank_user_satisfaction)
        rank_user_satisfaction.sort(key=lambda x: user_satisfaction[x], reverse=True)
        for i in range(m):
            next_user = rank_user_satisfaction[i]
    
            # find next item and provider
            for next_item in rec_flag[next_user]:
                next_provider_list = i_p[next_item]
                break_b=True
                for next_provider in next_provider_list:
                    if provider_exposure_score[next_provider] + 1 / math.log((top_k + 2), 2) > fair_exposure[next_provider]:
                        break_b=False
                if break_b:
                    rec_flag[next_user].remove(next_item)
                    break
    
            if next_item == sorted_score[next_user][-1]:
                continue
    
            rec_result[next_user][top_k] = next_item
            user_satisfaction[next_user] += \
                score[next_user][next_item] / math.log((top_k + 2), 2) / ideal_score[next_user]
            user_satisfaction_total += \
                score[next_user][next_item] / math.log((top_k + 2), 2) / ideal_score[next_user]
            for next_provider in next_provider_list:
                provider_exposure_score[next_provider] += 1 / math.log((top_k + 2), 2)
    rec_result = [[v+1 for v in r] for r in rec_result]
    with open('rcomendation_list'+str(q)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rec_result)
    return provider_exposure_score
def evaluate(p_d):
    lines={}
    for i in range(len(p_d)):
        count=0
        for j in p_d[i]:
            if j != 0:
                count+=1
        lines[i]=[i,str(count)+"/19"]
    for z in range(0,5):
        index_MRR = [0 for i in range(943)]
        result= pd.read_csv('rcomendation_list'+str(z)+'.csv', header=None)
        result= result.to_numpy().tolist()
        test= pd.read_csv('test'+str(z)+'.csv')
        listOfDFRows = test.to_numpy().tolist()
        listOfDFRows.sort(key = lambda i: i[0])
        users=[]
        l=0
        for j in range(1,944):
            user=[]
            for i in range(l,len(listOfDFRows)):
                if j==listOfDFRows[i][0]:
                    user.append(listOfDFRows[i][1])
                else:
                    l=i
                    break;
            users.append(user)
        NDCG_list=[]
        TotalNDCG=0
        for i in range(0,len(users)):
            min_index=1000
            NDCG=0
            for j in users[i]:
                if j in result[i]:
                    NDCG+=1
                    TotalNDCG+=1
                    if result[i].index(j) < min_index:
                        min_index=result[i].index(j) + 1
            index_MRR[i]=min_index
            NDCG_list.append(NDCG)
        lines[z].append(TotalNDCG/943)
        MRR=0
        for i in range(0,943):
            if index_MRR[i]!=1000:
                MRR+=1/index_MRR[i]
        lines[z].append(MRR/943)
    return lines
#__________________________________________________________
makingfileready()
five_foldCrossValid()
p_d=[]
for q in range(0,5):
    ratingMatrix(q)
    provider_exposure=reRank(q)
    p_d.append(provider_exposure)
lines=evaluate(p_d)
fields = ['test/train num','diversity', 'NDCG%', 'MRR'] 
with open('evaluation.csv', 'w',newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    for i in range(0,5):
        write.writerow(lines[i])
