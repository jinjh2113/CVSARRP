# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:40:06 2021
Wu method
@author: Lenovo
"""
import itertools
from itertools import chain
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.stats as stats
from scipy.stats import norm#拟合正态分布
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster
from sklearn import decomposition as skldec #用于主成分分析降维的包
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os
from networkx.algorithms import shortest_path_length
import random 
import json
newdotbl=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\dotbl20210729.xlsx",sheet_name='dotbl')

doidname=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\dotbl20210729.xlsx",sheet_name='doidname')
alldisease=doidname['DOID'].tolist()

doidtoicd10=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\ICD10CM_DOID.xlsx",sheet_name='ICD10CM_DOID')

doweighted14=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\do_weighted.xlsx",sheet_name='weighted14',index_col=0)
list14=doweighted14['DOID'].tolist()
doweighted33=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\do_weighted.xlsx",sheet_name='weighted33',index_col=0)
list33=doweighted33['DOID'].tolist()

doweighted474=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\do_weighted.xlsx",sheet_name='weighted474',index_col=0)
list474=doweighted474['DOID'].tolist()

mortality=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\data\\mortality.xlsx",sheet_name='Sheet1')
mortality.columns=['地区', '新增', '累计', '治愈', '死亡', 'mortality']
mortality.boxplot(['mortality'])#直接画箱线图
percentile = np.percentile(mortality['mortality'].tolist(), (25, 50, 75), interpolation='linear')

wuandpalmersimweightarray14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wuandpalmersimweightarray14.csv",index_col=0)
wuandpalmersimweightarray14.boxplot(['weightave'])
wusimweightarray14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wusimweightarray14.csv",index_col=0)
wusimweightarray14.boxplot(['weightave'])
wangsimweightarray14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimweightarray14.csv",index_col=0)
wangsimweightarray14.boxplot(['weightave14'])
wangsimweightarray33=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wang\\10791del\\simMatrix33.csv",index_col=0)
wangsimweightarray33.boxplot(['ave'])
wangsimweightarray260=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wang\\10791del\\simMatrix260.csv",index_col=0)
wangsimweightarray260.boxplot(['weightave'])

alldisease=list(set(newdotbl['do_id'].tolist())|set(newdotbl['parent'].tolist()))
alldisease.sort()
G=nx.DiGraph()#构建DO有相图（树图）
for i in range(len(alldisease)):
    G.add_node(alldisease[i], diseasename=doidname.loc[doidname['DOID']==alldisease[i],'DONAME'].tolist()[0])
for i in range(len(newdotbl)):
    G.add_edge(newdotbl.iloc[i,2],newdotbl.iloc[i,0])
nx.write_gexf(G,'E:\\博士生文件夹\\DO\\DO\\result\\network\\DOtree.gexf')

#生辰纲N个不同RGB颜色的列表     
def Colourlist_Generator(n):
    Rangelist = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    n = int(n)
    Colours = []             #空列表，用于插入n个表示颜色的字符串
    j = 1
    while j <= n:            #循环n次，每次在0到14间随机生成6个数，在加“#”符号，次次插入列表
        colour = ""          #空字符串，用于插入字符组成一个7位的表示颜色的字符串（第一位为#，可最后添加）
        for i in range(6):
            colour += Rangelist[random.randint(0,14)]    #用randint生成0到14的随机整数作为索引
        colour = "#"+colour                              #最后加上不变的部分“#”
        Colours.append(colour)
        j = j+1
    return Colours

#计算每个节点到根节点的距离
def getrootdic(do1,G):
    return nx.shortest_path_length(G, source='DOID:4', target=do1)

#计算两个节点的最近共同祖先节点（祖先节点到两个节点的距离之和最小）
def getMRCAnode(do1,do2,G):
    sources = [n for n, deg in G.in_degree if deg == 0]
    if len(sources) == 1:
        root = sources[0]#找到根节点
    if do1==root:
        ancestor1={root}
    else:
        path1=list(nx.all_simple_paths(G, source=root, target=do1))
        ancestor1=set(itertools.chain(*path1))#包含节点本身
    if do2==root:
        ancestor2={root}
    else:
        path2=list(nx.all_simple_paths(G, source=root, target=do2))
        ancestor2=set(itertools.chain(*path2))
    CA=list(ancestor1&ancestor2)
    if len(CA)==1:
        mrca=CA[0]
        mindic=nx.shortest_path_length(G, source=mrca, target=do1)+nx.shortest_path_length(G, source=mrca, target=do2)
    else:
        mrca=CA[0]
        mindic=nx.shortest_path_length(G, source=mrca, target=do1)+nx.shortest_path_length(G, source=mrca, target=do2)
        for i in CA[1:]:
            dic=nx.shortest_path_length(G, source=i, target=do1)+nx.shortest_path_length(G, source=i, target=do2)
            if dic<mindic:
                mrca=i
                mindic=dic
    return mrca,mindic#mindic为mrca到两个doi的最短路径长度和

#计算两个节点的最低共同邻祖先节点（祖先节点中层级最靠下的节点）
def getLCAnode(do1,do2,G):
    return nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(G, do1, do2)

#得到所有叶子节点8550个
def getleftnode(G):
    return [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)>=1]

#计算网络最大长度
def getdpG(G,leftnode):
    dicpath=nx.single_source_dijkstra_path_length(G, 'DOID:4')
    node={}
    for i in dicpath.keys():
        if i in leftnode:
            node[i]=dicpath[i]
    sortnode=sorted(node.items(), key = lambda kv:(kv[1], kv[0]))
    return sortnode[-1][1]
    
#计算每个节点到叶子节点的最短距离
def getleftdic(do1,G,leftnode):
    dicpath=nx.single_source_dijkstra_path_length(G, do1)
    node={}
    for i in dicpath.keys():
        if i in leftnode:
            node[i]=dicpath[i]
    sortnode=sorted(node.items(), key = lambda kv:(kv[1], kv[0]))
    return sortnode[0][1]
#计算任意两个节点之间的距离：(有向图)
def getdict(do1,do2,G):
    if nx.shortest_path_length(G, source='DOID:4', target=do1)>nx.shortest_path_length(G, source='DOID:4', target=do2):
        return nx.shortest_path_length(G, source=do2, target=do1)
    else:
        return nx.shortest_path_length(G, source=do1, target=do2)
#用MRCA算
def RSS(G,do1,do2):
    mrca,c=getMRCAnode(do1,do2,G)
    if mrca=='DOID:4':
        return 0
    a=getrootdic(mrca,G)
    leftnode=getleftnode(G)
    b=max(getleftdic(do1,G,leftnode),getleftdic(do2,G,leftnode))
    dp=getdpG(G,leftnode)#12
    rss=(dp*a)/((dp+c)*(a+b))
    return rss
#用LCA算
def RSS2(G,do1,do2):
    mrca=getLCAnode(do1,do2,G)
    if mrca=='DOID:4':
        return 0
    a=getrootdic(mrca,G)
    leftnode=getleftnode(G)
    b=max(getleftdic(do1,G,leftnode),getleftdic(do2,G,leftnode))
    c=nx.shortest_path_length(G, source=mrca, target=do1)+nx.shortest_path_length(G, source=mrca, target=do2)
    dp=getdpG(G,leftnode)#12
    rss=(dp*a)/((dp+c)*(a+b))
    return rss
#wu和palmer的求相似性方法：
def wuandpalmersim(G,do1,do2):
    lca=getLCAnode(do1,do2,G)
    if lca=='DOID:4':
        return 0
    n3=getrootdic(lca,G)
    n1=nx.shortest_path_length(G, source=lca, target=do1)
    n2=nx.shortest_path_length(G, source=lca, target=do2)
    sim=(2*n3)/(n1+n2+2*n3)
    return sim

def getRSSsimlist():
    similarity=[]
    name=['DOID1','DOID2','similarity']
    list14=['DOID:114', 'DOID:1307', 'DOID:557', 'DOID:863', 'DOID:1579', 'DOID:162', 'DOID:9351', 'DOID:409', 'DOID:10763', 'DOID:2841', 'DOID:8893', 'DOID:7148', 'DOID:8857', 'DOID:2531']
    for i in range(len(alldisease)):
        for j in range(len(list14)):
                similarity.append([alldisease[i],list14[j],RSS(G,alldisease[i],list14[j])])
        print(i)
        if (i>2000)&(i%2000==0):
            dfi=pd.DataFrame(columns=name,data=similarity)
            dfi.to_csv("E:\\博士生文件夹\\DO\\network\\wusim14_{}.csv".format(i))
        
    df=pd.DataFrame(columns=name,data=similarity)
    df.to_csv("E:\\博士生文件夹\\DO\\network\\wusim14.csv")
    
def getwuandpsimlist():
    similarity=[]
    list14=['DOID:114', 'DOID:1307', 'DOID:557', 'DOID:863', 'DOID:1579', 'DOID:162', 'DOID:9351', 'DOID:409', 'DOID:10763', 'DOID:2841', 'DOID:8893', 'DOID:7148', 'DOID:8857', 'DOID:2531']
    for i in range(len(alldisease)):
        for j in range(len(list14)):
                similarity.append([alldisease[i],list14[j],wuandpalmersim(G,alldisease[i],list14[j])])
        print(i)
    name=['DOID1','DOID2','similarity']
    df=pd.DataFrame(columns=name,data=similarity)
    df.to_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmersim14.csv")
    
#将几个分开数据合并
def concatdata1():
    df14_1=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersim14_1.csv",index_col=0)
    df14_2=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersim14_2.csv",index_col=0)
    df14_3=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersim14_3.csv",index_col=0)
    
    df14=df14_1.copy()
    df14_2.index=range(42000,84000)
    df14=pd.concat([df14_1,df14_2],axis=0,join='outer',sort=False)
    df14_3.index=range(84000,151970)
    df14=pd.concat([df14,df14_3],axis=0,join='outer',sort=False)
    df14.to_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersim14.csv")
def concatdata2():
    df14_1=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14_0_2000.csv",index_col=0)
    df14_2=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14_2000_4000.csv",index_col=0)
    df14_3=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14_4000_6000.csv",index_col=0)
    df14_4=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14_6000_8000.csv",index_col=0)
    df14_5=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14_8000_10855.csv",index_col=0)
    
    df14=df14_1.copy()
    df14_2.index=range(28000,56000)
    df14=pd.concat([df14_1,df14_2],axis=0,join='outer',sort=False)
    df14_3.index=range(56000,84000)
    df14=pd.concat([df14,df14_3],axis=0,join='outer',sort=False)
    df14_4.index=range(84000,112000)
    df14=pd.concat([df14,df14_4],axis=0,join='outer',sort=False)
    df14_5.index=range(112000,151970)
    df14=pd.concat([df14,df14_5],axis=0,join='outer',sort=False)
    df14.to_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14.csv")

def concatdata3():
    df14_1=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim0_300.csv",index_col=0)#3211650
    df14_2=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim300_600.csv",index_col=0)#3121650
    df14_3=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim600_900.csv",index_col=0)#3031650
    df14_4=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim900_1200.csv",index_col=0)#2941650
    df14_5=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim1200_1500.csv",index_col=0)#2851650
    df14_6=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim1500_1700.csv",index_col=0)#1851100
    df14_7=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim1700_2000.csv",index_col=0)#2701650
    df14_8=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim2000_3000.csv",index_col=0)#8355500
    df14_9=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim3000_3500.csv",index_col=0)#3802750
    df14_10=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim3500_4000.csv",index_col=0)#3552750
    df14_11=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim4000_4600.csv",index_col=0)#3933300
    df14_12=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim4600_5000.csv",index_col=0)#2422200
    df14_13=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim5000_5500.csv",index_col=0)#2802750
    df14_14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim5500_6000.csv",index_col=0)#2552750
    df14_15=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\sim6000_10855.csv",index_col=0)#11787940

    df14=df14_1.copy()
    df14_2.index=range(3211650,6333300)
    df14=pd.concat([df14_1,df14_2],axis=0,join='outer',sort=False)
    df14_3.index=range(6333300,9364950)
    df14=pd.concat([df14,df14_3],axis=0,join='outer',sort=False)
    df14_4.index=range(9364950,12306600)
    df14=pd.concat([df14,df14_4],axis=0,join='outer',sort=False)
    df14_5.index=range(12306600,15158250)
    df14=pd.concat([df14,df14_5],axis=0,join='outer',sort=False)
    df14_6.index=range(15158250,17009350)
    df14=pd.concat([df14,df14_6],axis=0,join='outer',sort=False)
    df14_7.index=range(17009350,19711000)
    df14=pd.concat([df14,df14_7],axis=0,join='outer',sort=False)
    df14_8.index=range(19711000,28066500)
    df14=pd.concat([df14,df14_8],axis=0,join='outer',sort=False)
    df14_9.index=range(28066500,31869250)
    df14=pd.concat([df14,df14_9],axis=0,join='outer',sort=False)
    df14_10.index=range(31869250,35422000)
    df14=pd.concat([df14,df14_10],axis=0,join='outer',sort=False)
    df14_11.index=range(35422000,39355300)
    df14=pd.concat([df14,df14_11],axis=0,join='outer',sort=False)
    df14_12.index=range(39355300,41777500)
    df14=pd.concat([df14,df14_12],axis=0,join='outer',sort=False)
    df14_13.index=range(41777500,44580250)
    df14=pd.concat([df14,df14_13],axis=0,join='outer',sort=False)
    df14_14.index=range(44580250,47133000)
    df14=pd.concat([df14,df14_14],axis=0,join='outer',sort=False)
    df14_15.index=range(47133000,58920940)
    df14=pd.concat([df14,df14_15],axis=0,join='outer',sort=False)
    df14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\networksim.csv")
    #将三列的数据转为对称矩阵形式
    arr=np.zeros((10855,10855))
    n_data = pd.DataFrame(arr,columns = alldisease,index=alldisease)
    for i in range(len(df14)):
        do1=df14.loc[i,'DOID1']
        do2=df14.loc[i,'DOID2']
        similarity=df14.loc[i,'similarity']
        if do1==do2:
            n_data.loc[do1,do2]=similarity
        else:
            n_data.loc[do1,do2]=similarity
            n_data.loc[do2,do1]=similarity
    n_data.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\networksimarray.csv")



#将列表[doid1,doid2,sim]转为对称矩阵形式
def dftoarray():
    df14=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusim14.csv",index_col=0)
    arr = np.zeros((10855,14))
    n_data = pd.DataFrame(arr,columns = list14,index=alldisease)
    for i in range(len(df14['similarity'].tolist())):
        n_data.loc[df14.loc[i,'DOID1'],df14.loc[i,'DOID2']]=df14.loc[i,'similarity']
    n_data.to_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusimarray14.csv")

#对矩阵进行加权求平均    
def getweighted():
    n_data=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusimarray14.csv",index_col=0)
    weightave14=[0]*len(alldisease)
    for i in range(len(alldisease)):
        weight=0
        for j in range(len(list14)):
            weight+=n_data.loc[alldisease[i],list14[j]]*float(doweighted14.loc[list14[j],'weight'])
        weightave14[i]=weight/14
    n_data['weightave14']=weightave14
    n_data.to_csv("E:\\博士生文件夹\\DO\\network\\wu\\wusimweightarray14.csv")
    
    n_data=pd.read_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersimarray14.csv",index_col=0)
    weightave14=[0]*len(alldisease)
    for i in range(len(alldisease)):
        weight=0
        for j in range(len(list14)):
            weight+=n_data.loc[alldisease[i],list14[j]]*float(doweighted14.loc[list14[j],'weight'])
        weightave14[i]=weight/14
    n_data['weightave14']=weightave14
    n_data.to_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmer\\wuandpalmersimweightarray14.csv")
    
    #list33的权重均为1直接平均
    df33=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimarray33.csv",index_col=0)
    dfave33=df33.copy()
    dfave33['ave33']=df33.apply(lambda x:x.mean(),axis=1)
    dfave33.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimweightarray33.csv")
    
    df474=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimarray474.csv",index_col=0)
    weightave474=[0]*len(alldisease)
    for i in range(len(alldisease)):
        weight=0
        for j in range(len(list474)):
            weight+=df474.loc[alldisease[i],list474[j]]*float(doweighted474.loc[list474[j],'weight'])
        weightave474[i]=weight/474
    df474['weightave474']=weightave474
    df474.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimweightarray474.csv")
    
#得到父子节点和邻居节点
def getadjnode():
    parent14=set(newdotbl.loc[newdotbl['do_id'].isin(list14),['parent']]['parent'].tolist())#13个
    son14=set(newdotbl.loc[newdotbl['parent'].isin(list14),['do_id']]['do_id'].tolist())#107个
    adj14=set(parent14|son14|set(list14))#133个
    print("parent14:",len(parent14))
    print("son14:",len(son14))
    print("adj14:",len(adj14))
    
    parent33=set(newdotbl.loc[newdotbl['do_id'].isin(list33),['parent']]['parent'].tolist())#29个
    son33=set(newdotbl.loc[newdotbl['parent'].isin(list33),['do_id']]['do_id'].tolist())#223个
    adj33=set(parent33|son33|set(list33))#276个
    print("parent33:",len(parent33))
    print("son33:",len(son33))
    print("adj33:",len(adj33))
    
    parent474=set(newdotbl.loc[newdotbl['do_id'].isin(list474),['parent']]['parent'].tolist())#200个
    son474=set(newdotbl.loc[newdotbl['parent'].isin(list474),['do_id']]['do_id'].tolist())#889个
    adj474=set(parent474|son474|set(list474))#1270个
    print("parent474:",len(parent474))
    print("son474:",len(son474))
    print("adj474:",len(adj474))
    
    adjall=set(adj14|adj33|adj474)#1426个
    print('alladj:',adjall)
    print('len(set(list474)|set(list33)|set(list14)):',len(set(list474)|set(list33)|set(list14)))
    print('len(set(parent474)|set(parent33)|set(parent14)):',len(set(parent474)|set(parent33)|set(parent14)))
    print('len(set(son474)|set(son33)|set(son14)):',len(set(son474)|set(son33)|set(son14)))
    print('len(set(adj474)|set(adj33)|set(adj14)):',len(set(adj474)|set(adj33)|set(adj14)))
    print('len(parent14&set(list474))',len(parent14&set(list474)))
    print('len(parent33&set(list474))',len(parent33&set(list474)))
    print('len(son14&set(list474))',len(son14&set(list474)))
    print('len(son33&set(list474))',len(son33&set(list474)))
    print('son14&parent14',son14&parent14)
    print('son33&parent33',son33&parent33)
    return adjall,parent14,son14,adj14,parent33,son33,adj33,parent474,son474,adj474

adjall,parent14,son14,adj14,parent33,son33,adj33,parent474,son474,adj474=getadjnode()
#按照邻居节点计算分值
#对wangweightave14进行修改
def getadjweightscore():
    #王的相似性矩阵
    df10855=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\networkwangsimarray.csv",index_col=0)
    #计算的分值
    dfallscore=pd.read_excel("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\prescoreall.xlsx",index_col=0)
    df=dfallscore.loc[adjall,:]
    for i in df.index:
        if i in list14:
            df.loc[i,'wangweightave14']=doweighted14.loc[i,'weight']
        elif i in parent14 or i in son14:
            parent=newdotbl.loc[newdotbl['do_id']==df.iloc[i,0]]['parent'].tolist()
            son=newdotbl.loc[newdotbl['parent']==df.iloc[i,0]]['do_id'].tolist()
            adj=parent+son
            df.loc[i,'wangweightave14']=0
            for node in adj:
                if node in list14:
                    df.loc[i,'wangweightave14']+=df10855.loc[i,node]*doweighted14.loc[node,'weight']
            df.loc[i,'wangweightave14']=format(df.loc[i,'wangweightave14']/len(adj), ".4f")+'*'
        else:
            df.loc[i,'wangweightave14']='-'
    df.to_csv('E:\\博士生文件夹\\DO\\DO\\result\\prescore\\prescore14adj.csv',encoding='gbk')

#所有非0的值加权求和，第一种分值预测方法中的wu和wuandp方法的计算方式
def getweight():
    wudf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\wusimweightarray14.csv",index_col=0)
    for i in wudf14.index:
        score=0
        count=0
        for j in wudf14.columns:
            nowvalue=wudf14.loc[i,j]
            if nowvalue!=0:
                score+=nowvalue
                count+=1
        if count==0:
            wudf14.loc[i,'weightave']=0
        else:
            wudf14.loc[i,'weightave']=score/count
    wudf14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\wusimweightarray14_2.csv")
    
#计算所有分值（考虑一下可以删除）
def getallscore():
    wpdf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wuandpalmersimweightarray14.csv",index_col=0)
    wangdf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wangsimweightarray14.csv",index_col=0)
    wudf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\wusimweightarray14.csv",index_col=0)
    data={'DONAME':doidname['DONAME'].tolist(),'wangscore14':wangdf14['weightave'].tolist(),'wuscore14':wudf14['weightave'].tolist(),'wpscore14':wpdf14['weightave'].tolist()}
    index=alldisease
    df14=pd.DataFrame(data=data,index=index)
    df14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\prescore14.csv")
    #将DOID与ICD10CM对应
    mergedf14=pd.merge(df14,doidtoicd10,on='DOID',how='inner')
    mergedf14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\prescore14_ICD3560.csv")
    #合并王老师数据，找到对应ICD10CM的疾病
    WangIllness=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\data\\WangIllness.csv",index_col=0)
    ICD10CM=list(set(doidtoicd10['ICD10CM'].tolist()))
    Wangdata=WangIllness.loc[WangIllness['ICD10CM'].isin(ICD10CM)]#872个
    Wangdata=Wangdata.loc[(Wangdata['total']>=15)&(Wangdata['ratio']>=0.03)]#211个
    DOID=[]
    for i in Wangdata['ICD10CM'].tolist():
        DOID.append(doidtoicd10.loc[doidtoicd10['ICD10CM']==i]['DOID'].tolist())
    Wangdata['DOID']=DOID
    Wangdata.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\Wangdata211.csv")
    DOID474=list(set(list(itertools.chain(*DOID)))).sort()#474个
    weightedlist=[]
    for i in Wangdata.index:
        for j in Wangdata.loc[i,'DOID']:
            weightedlist.append([j,Wangdata.loc[i,'ICD10CM'],Wangdata.loc[i,'ratio']])
    #有三个重复的求平均，手动
    
    merge14=pd.merge(mergedf14,WangIllness,on='ICD10CM',how='inner')#1691个
    merge14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\prescore14_ICD_WangIllness1691.csv")

#根据相似度矩阵构建网络
adjall=getadjnode()
alladjdisease=list(adjall)
alladjdisease.sort()
def createsimnet(alladjdisease):
    n_data=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\networkwangsimarray.csv",index_col=0)
    subnetdf=n_data.loc[alladjdisease,alladjdisease]
    subG=nx.Graph()
    for i in range(len(alladjdisease)):
        subG.add_node(alladjdisease[i], diseasename=doidname.loc[doidname['DOID']==alladjdisease[i],'DONAME'].tolist()[0])
    for i in range(len(alladjdisease)):
        for j in range(i+1,len(alladjdisease)):
            if subnetdf.loc[alladjdisease[i],alladjdisease[j]]>0.6:#边的权重可以修改
                subG.add_edge(alladjdisease[i],alladjdisease[j],weight=subnetdf.loc[alladjdisease[i],alladjdisease[j]])
    nx.write_gexf(subG,'E:\\博士生文件夹\\DO\\DO\\result\\network\\wang1426_6.gexf')
    print("网络边个数：",len(subG.edges()))

##创建全部的图,不同类别的节点用不同的颜色
def createTree(G,node,nodecolors,result_dic):
    #创建树
    children=list(G[node].keys())#获得node节点的所有孩子节点
    if node in result_dic.keys():
        myTree = {"name":G.nodes[node]['diseasename']+"\n"+node,"itemStyle":{"color": nodecolors[result_dic[node]] },"children":[],"value":result_dic[node]}
    else:
        myTree = {"name":G.nodes[node]['diseasename']+"\n"+node,"itemStyle":{"color": nodecolors[-1] },"children":[],"value":18}
    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree（）
    for node in children:
        myTree["children"].append (createTree(G,node,nodecolors,result_dic))
    return myTree

#节点标签为疾病名和doid
def createTree2(G,node,nodecolors,result_dic):
    #创建树
    children=list(G[node].keys())#获得node节点的所有孩子节点
    if node in result_dic.keys():
        myTree = {"name":G.nodes[node]['diseasename']+"\n"+node,"itemStyle":{"color": nodecolors[result_dic[node]] },"children":[],"value":result_dic[node]}
    else:
        myTree = {"name":G.nodes[node]['diseasename']+"\n"+node,"itemStyle":{"color": nodecolors[-1] },"children":[],"value":18}
    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree（）
    for node in children:
        myTree["children"].append (createTree2(G,node,nodecolors,result_dic))
    return myTree

def createTree3(G,node,nodecolors,result_dic):
    #创建树
    children=list(G[node].keys())#获得node节点的所有孩子节点
    if node in result_dic.keys():
        myTree = {"name":G.nodes[node]['diseasename'],"itemStyle":{"color": nodecolors[result_dic[node]] },"children":[],"value":result_dic[node]}
    else:
        myTree = {"name":G.nodes[node]['diseasename'],"itemStyle":{"color": nodecolors[-1] },"children":[],"value":18}
    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree（）
    for node in children:
        myTree["children"].append (createTree3(G,node,nodecolors,result_dic))
    return myTree


#将某个gephi分好的模块节点数据输入，导入到DO中，输出树图json，导入到echarts中，看划分情况。
def getechartstree():
    nodesclass=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\1426_4nodes.csv")
    #key唯一时可以直接转为字典形式
    result_dic = nodesclass.groupby('Id')['modularity_class'].apply(lambda x:int(x)).to_dict()
    nodecolors=Colourlist_Generator(19)#生成表示18个不同rgb颜色的列表
    nodecolors=['#5BA248', '#A732E9', '#F8225B', '#DDA3B9', '#C9A2B9', '#4D72E9', '#8B9621', '#951A94', '#E6486B', '#A5FEEE', '#1335FC', '#A5F374', '#98D38B', '#C6127C', '#EFAF53', '#559BC1', '#CC9748', '#78AC83', '#98EB96']
    nodecolors=['#FF0000', '#A732E9', '#FF851B', '#A5F374', '#C282A9', '#4D72E9', '#8B9621', '#951A94', '#A5FEEE', '#85144B', '#1335FC', '#ff6600', '#FFDC00', '#C6127C', '#ffff00', '#559BC1', '#CC9748', '#ff0066', '#AAAAAA']
    tree=createTree3(G,'DOID:4',nodecolors,result_dic)
    with open('E:\\博士生文件夹\\DO\\DO\\result\\network\\echart_tree_DO_classes.json','a',encoding='utf8')as fp:
        json.dump(tree,fp,ensure_ascii=False)#将字典的形式转为json格式(单引号变为双引号)

#将gephi的各个节点标签隐藏，只保留14个节点的标签，并改变节点的大小
def getgephinodes():
    nodesclass2=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\1426_4nodes.csv")
    nodesclass2['Label']=[np.nan]*1426
    nodesclass2['size']=[1]*1426
    for i in list14:
        nodesclass2.loc[nodesclass2['Id']==i,'Label']=nodesclass2.loc[nodesclass2['Id']==i]['0']
        nodesclass2.loc[nodesclass2['Id']==i,'size']=2
    nodesclass2.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\1426_4nodes_labelsize.csv")
    
    
#层次聚类，将相似度作为距离
def hcluster():
    n_data=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\networkwangsimarray.csv",index_col=0)
    subnetdf=n_data.loc[alladjdisease,alladjdisease]

alladj=list(adjall)
import glob
#将csv文件直接合并
def gethebing():
    csv_list=glob.glob("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\*.csv")
    for i in csv_list:
        fr=open(i,'rb').read()
        with open('E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wuresult.csv','ab') as f:
            f.write(fr)
    #读取计算好的相似度网络三列数据
    df=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wuresult.csv",header=None,index_col=False,encoding='utf-8-sig')#注意这块的编码方式
    df.columns=['DOID1','DOID2','similarity']
    #将三列的数据转为对称矩阵形式
    arr=np.zeros((10855,10855))
    n_data = pd.DataFrame(arr,columns = alldisease,index=alldisease)
    for i in range(len(df)):
        do1=df.loc[i,'DOID1']
        do2=df.loc[i,'DOID2']
        similarity=df.loc[i,'similarity']
        if do1==do2:
            n_data.loc[do1,do2]=similarity
        else:
            n_data.loc[do1,do2]=similarity
            n_data.loc[do2,do1]=similarity
    n_data.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wunetworksimarray.csv")
    
    #将1426个邻居节点拖出来，构建网络
    df1426=df.loc[(df['DOID1'].isin(adjall))&(df['DOID2'].isin(adjall))]
    df1426.index=range(len(df1426))
    #将三列的数据转为对称矩阵形式
    arr=np.zeros((1426,1426))
    n_data = pd.DataFrame(arr,columns = alladj,index=alladj)
    for i in range(len(df1426)):
        do1=df1426.loc[i,'DOID1']
        do2=df1426.loc[i,'DOID2']
        similarity=df1426.loc[i,'similarity']
        if do1==do2:
            n_data.loc[do1,do2]=similarity
        else:
            n_data.loc[do1,do2]=similarity
            n_data.loc[do2,do1]=similarity
    n_data.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wunetworksimarray1426.csv")

#根据网络划分进行分值计算
def getnewclassscore():
    df1426_4=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\1426_4nodes.csv")
    list14_class=df1426_4.loc[df1426_4['Id'].isin(list14),['Id','modularity_class']]
    list902=df1426_4.loc[df1426_4['modularity_class'].isin(list14_class['modularity_class'].tolist())]['Id'].tolist()
    dfwang=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\networkwangsimarray.csv",index_col=0)
    dfwang14=dfwang.loc[list902,list14]
    prescore=[]
    for i in dfwang14.index:
        classi=df1426_4.loc[df1426_4['Id']==i,'modularity_class'].tolist()[0]
        samelist14=list14_class.loc[list14_class['modularity_class']==classi]['Id'].tolist()
        score=0
        count=0
        for j in samelist14:
            score+=dfwang14.loc[i,j]*doweighted14.loc[j,'weight']
            count+=dfwang14.loc[i,j]
        prescore.append(score/count)
    dfwang14['prescore']=prescore
    doname=[]
    for i in dfwang14.index:
        doname.append(doidname.loc[doidname['DOID']==i]['DONAME'].tolist()[0])
    dfwang14['doname']=doname
    dfwang14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\prescore902.csv")
    
    df1426_4=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wuandp\\wuandp1426_4nodes.csv")
    list14_class=df1426_4.loc[df1426_4['Id'].isin(list14),['Id','modularity_class']]
    list904=df1426_4.loc[df1426_4['modularity_class'].isin(list14_class['modularity_class'].tolist())]['Id'].tolist()
    dfwang=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wuandp\\wuandpnetworksimarray1426.csv",index_col=0)
    dfwang14=dfwang.loc[list904,list14]
    prescore=[]
    for i in dfwang14.index:
        classi=df1426_4.loc[df1426_4['Id']==i,'modularity_class'].tolist()[0]
        samelist14=list14_class.loc[list14_class['modularity_class']==classi]['Id'].tolist()
        score=0
        count=0
        for j in samelist14:
            score+=dfwang14.loc[i,j]*doweighted14.loc[j,'weight']
            count+=dfwang14.loc[i,j]
        if count==0:
            prescore.append(0)
        else:
            prescore.append(score/count)
    dfwang14['prescore']=prescore
    doname=[]
    for i in dfwang14.index:
        doname.append(doidname.loc[doidname['DOID']==i]['DONAME'].tolist()[0])
    dfwang14['doname']=doname
    dfwang14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wuandp\\wuandpprescore904.csv")
    
    df1426_4=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wu1426_6nodes.csv")
    list14_class=df1426_4.loc[df1426_4['Id'].isin(list14),['Id','modularity_class']]
    list992=df1426_4.loc[df1426_4['modularity_class'].isin(list14_class['modularity_class'].tolist())]['Id'].tolist()
    dfwang=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wunetworksimarray1426.csv",index_col=0)
    dfwang14=dfwang.loc[list992,list14]
    prescore=[]
    for i in dfwang14.index:
        classi=df1426_4.loc[df1426_4['Id']==i,'modularity_class'].tolist()[0]
        samelist14=list14_class.loc[list14_class['modularity_class']==classi]['Id'].tolist()
        score=0
        count=0
        for j in samelist14:
            score+=dfwang14.loc[i,j]*doweighted14.loc[j,'weight']
            count+=dfwang14.loc[i,j]
        prescore.append(score/count)
    dfwang14['prescore']=prescore
    doname=[]
    for i in dfwang14.index:
        doname.append(doidname.loc[doidname['DOID']==i]['DONAME'].tolist()[0])
    dfwang14['doname']=doname
    dfwang14.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wuprescore992.csv")
    
    #三组数据加权求和
    listall=list(set(list902)&set(list904)&set(list992))#839个
    dfwang902=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wang\\prescore902.csv",index_col=0)
    dfwuandp904=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wuandp\\wuandpprescore904.csv",index_col=0)
    dfwu992=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\wu\\wuprescore992.csv",index_col=0)
    
    dfwang839=dfwang902.loc[listall,['prescore','doname']]
    dfwang839.columns=['prescorewang','doname']
    dfwuandp839=dfwuandp904.loc[listall,['prescore','doname']]
    dfwuandp839.columns=['prescorewuandp','doname']
    dfwu839=dfwu992.loc[listall,['prescore','doname']]
    dfwu839.columns=['prescorewu','doname']
    dfall839=pd.concat([dfwang839,dfwuandp839,dfwu839],axis=1,join='outer')
    
    dfall=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\dfall839.csv",index_col=0)
    dfallcrawlinf1=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\crawldata\\table\\allcrawlinf1.csv")
    dfallcrawlinf1.index=dfallcrawlinf1.loc['DOID'].tolist()
    dfallresult=pd.concat([dfall,dfallcrawlinf1],axis=1,join='inner')
    dfallresult.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\network\\dfallresult839.csv")
    #加上之前算过的第一种预测方法
    wangdf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\wangsimweightarray14.csv",index_col=0)
    #wang的方法需要归一化处理
    wangdf14=wangdf14.loc[:,['weightave14']]
    minvalue=min(wangdf14['weightave14'].tolist())
    maxvalue=max(wangdf14['weightave14'].tolist())
    for i in wangdf14.index:
        wangdf14.loc[i,'weightave14']=(wangdf14.loc[i,'weightave14']-minvalue)*0.62/(maxvalue-minvalue)
    wangdf14.columns=['globalprescorewang']
    wudf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\wusimweightarray14_2.csv",index_col=0)
    wudf14=wudf14.loc[:,['weightave']]
    wudf14.columns=['globalprescorewu']
    wuandpdf14=pd.read_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\wuandpalmersimweightarray14_2.csv",index_col=0)
    wuandpdf14=wuandpdf14.loc[:,['weightave']]
    wuandpdf14.columns=['globalprescorewuandp']
    #全局分数加上局部分数求和
    dfall10855=pd.concat([wangdf14,wudf14,wuandpdf14],axis=1,join='inner')#需要乘以各自算法的比重
    df839=dfallresult.loc[:,['doname', 'prescorewang', 'prescorewuandp', 'prescorewu', 'ave','allresult']]
    dfglobalall839=pd.concat([df839,dfall10855],axis=1,join='inner')
    score=[]
    globalscore=[]
    for i in dfglobalall839.index:
        scorei=dfglobalall839.loc[i,'globalprescorewang']*32*28/(32*28+42*32+42*28)+dfglobalall839.loc[i,'globalprescorewuandp']*42*28/(32*28+42*32+42*28)+dfglobalall839.loc[i,'globalprescorewu']*42*32/(32*28+42*32+42*28)
        globalscore.append(scorei)
        score.append(scorei*0.5+dfglobalall839.loc[i,'ave']*0.5)
        
    dfglobalall839['globalscore']=globalscore
    dfglobalall839['score']=score
    dfglobalall839.to_csv("E:\\博士生文件夹\\DO\\DO\\result\\prescore\\14\\dfglobalall839.csv")
    
    #计算R2
    from sklearn.linear_model import LinearRegression
    a=dfglobalall839.loc[list14,['score']].sort_values(by='score')
    df14R2=pd.concat([a,doweighted14],axis=1,sort=False,join='inner')
    x=df14R2['weight'].tolist()
    y=df14R2['score'].tolist()
    z=df14R2['globalscore'].tolist()
    w=df14R2['ave'].tolist()
    x1=pd.Series(x)
    y1=pd.Series(y)
    z1=pd.Series(z)
    w1=pd.Series(w)
    corr=round(x1.corr(y1),4)#标准差R0.9703
    corrglobalscore=round(x1.corr(z1),4)#0.8461
    corrlocalscore=round(x1.corr(w1),4)#0.9658
    
    #
    
if __name__ == '__main__': 
#    similarity=[]
#    list14=['DOID:114', 'DOID:1307', 'DOID:557', 'DOID:863', 'DOID:1579', 'DOID:162', 'DOID:9351', 'DOID:409', 'DOID:10763', 'DOID:2841', 'DOID:8893', 'DOID:7148', 'DOID:8857', 'DOID:2531']
#    for i in range(2000):
#        for j in range(len(list14)):
#                similarity.append([alldisease[i],list14[j],wuandpalmersim(G,alldisease[i],list14[j])])
#        print(i)
#                
#    name=['DOID1','DOID2','similarity']
#    df=pd.DataFrame(columns=name,data=similarity)
#    df.to_csv("E:\\博士生文件夹\\DO\\network\\wuandpalmersim14_1.csv")
    similarity=[]
    name=['DOID1','DOID2','similarity']
    list14=['DOID:114', 'DOID:1307', 'DOID:557', 'DOID:863', 'DOID:1579', 'DOID:162', 'DOID:9351', 'DOID:409', 'DOID:10763', 'DOID:2841', 'DOID:8893', 'DOID:7148', 'DOID:8857', 'DOID:2531']
    for i in range(len(alldisease)):
        for j in range(len(list14)):
                similarity.append([alldisease[i],list14[j],RSS(G,alldisease[i],list14[j])])
        print(i)
        if (i>2000)&(i%2000==0):
            dfi=pd.DataFrame(columns=name,data=similarity)
            dfi.to_csv("E:\\博士生文件夹\\DO\\network\\wusim14_{}.csv".format(i))
        
    df=pd.DataFrame(columns=name,data=similarity)
    df.to_csv("E:\\博士生文件夹\\DO\\network\\wusim14.csv")
