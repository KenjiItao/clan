import numpy as np
import pandas as pd
import random
import sys
import os
import csv
import collections
import copy
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pyclustering.cluster import xmeans
# import seaborn as sns
# sns.set_style(style="whitegrid")
# from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

if int(sys.argv[1])==0:
    if not os.path.exists("./cluster"):
        os.mkdir("./cluster")
    if not os.path.exists("./incest"):
        os.mkdir("./incest")

def cluster(x,y):
    try:
        init_center = xmeans.kmeans_plusplus_initializer(np.c_[x,y], 2).initialize()
        xm = xmeans.xmeans(np.c_[x,y], init_center, ccore=False)
        xm.process()
        sizes = [len(cluster) for cluster in xm.get_clusters()]
        centers=xm.get_centers()

        # cluster=x_means.x_means_cluster()
        # centers=cluster.cluster_centers_
        num_clans=len(centers)
        clans=[]
        for i in range(num_clans):
            mate=0
            cur=100
            for j in range(num_clans):
                if abs(centers[i][1]-centers[j][0])<cur:
                    mate=j
                    cur=abs(centers[i][1]-centers[j][0])
            clans.append([i,mate])
        candidate=list(range(num_clans))
        cur_cycle=0
        while len(candidate)>0:
            cycle=[candidate[0]]
            cur=candidate[0]
            while True:
                next=clans[cur][1]
                if clans[cur][1] in cycle:
                    if cur_cycle<len(cycle)-cycle.index(next):
                        cur_cycle=len(cycle)-cycle.index(next)
                    for clan in cycle:
                        if clan in candidate:
                            candidate.remove(clan)
                    break
                else:
                    cycle.append(next)
                    cur=next
    except:
        cur_cycle=1
    return cur_cycle

class Village:
    def __init__(self):
        self.clans=[]
        self.population=0

class Clan:
    def __init__(self,value,destiny,num_couple):
        self.value=value
        self.woman_value=destiny
        self.couple=round(num_couple)
        self.man=0
        self.woman=0
        self.candidate=[]

def year(vill):
    vill.population=0
    clans=[clan for clan in vill.clans if clan.couple>0]
    value_ls=np.array([clan.value for clan in clans])
    woman_value_ls=np.array([clan.woman_value for clan in clans])
    for clan in clans:
        distance=np.array([(clan.value-value_ls)**2,(clan.value-woman_value_ls)**2,(clan.woman_value-value_ls)**2]).min(axis=0)
        friend=np.sum(np.exp(-distance))/len(clans)
        rate=1/(1+coop*(1-friend))
        couple=birth*clan.couple
        clan.man=round(np.random.poisson(lam=couple)*rate)
        clan.woman=round(np.random.poisson(lam=couple)*rate)
        clan.couple=0
    for clan in clans:
        if clan.man+clan.woman>4*initial_pop:
            n=math.floor(math.log2((clan.man+clan.woman)/2/initial_pop))
            clan.man=round(clan.man/2**n)
            clan.woman=round(clan.woman/2**n)
            for i in [0]*(2**n-1):
                clans.append(Clan(clan.value,clan.woman_value,0))
                clans[-1].man=clan.man
                clans[-1].woman=clan.woman
    clans=[clan for clan in clans if clan.man*clan.woman>0]
    for clan in clans:
        clan.value+=mutation*(2*random.random()-1)
        clan.woman_value+=mutation*(2*random.random()-1)
    woman_value_ls=np.array([clan.woman_value for clan in clans])
    for clan in clans:
        enemy=np.sum(np.exp(-(clan.woman_value-woman_value_ls)**2))/len(clans)
        rate=1/(1+conflict*enemy)
        clan.man=round(clan.man*rate)
        clan.woman=round(clan.woman*rate)
        vill.population+=clan.man+clan.woman
    vill.clans=clans

def mating(vill):
    clans=vill.clans
    value_ls=np.array([clan.woman_value for clan in clans])
    for clan in clans:
        if clan.man>0:
            mates=np.exp(-(clan.value-value_ls)**2)
            w2 = mates / np.sum(mates)
            mate = np.random.choice(clans, p=w2)
            mate.candidate.append(clan)
    for mate in clans:
        if mate.woman<1 or len(mate.candidate)==0:
            mate.candidate=[]
            continue
        random.shuffle(mate.candidate)
        for lineage in mate.candidate:
            if mate.woman<1:
                break
            couple=min(lineage.man,mate.woman)
            lineage.man-=couple
            mate.woman-=couple
            lineage.couple+=couple
        mate.candidate=[]

def main():
    vills=[]
    num=0
    initial_population=initial_pop*num_lineage*2
    for i in range(num_vills):
        vills.append(Village())
        for j in range(num_lineage):
            vills[i].clans.append(Clan(random.gauss(0,1),random.gauss(0,1),initial_pop))

    cycles=[]
    incests=[]
    while num <500:
        remove_ls=[]
        duplicate_ls=[]
        for vill in vills:
            year(vill)
            if vill.population<initial_population/10:
                remove_ls.append(vill)
            elif vill.population>initial_population*2:
                duplicate_ls.append(vill)
        for vill in remove_ls:
            vills.remove(vill)
        for vill in duplicate_ls:
            random.shuffle(vill.clans)
            n=math.floor(math.log2(vill.population/initial_population))
            k=round(len(vill.clans)/2**n)
            for i in [0]*(2**n-1):
                clans=vill.clans[:k]
                vills.append(Village())
                vills[-1].clans=copy.deepcopy(clans)
                vill.clans=vill.clans[k:]
        if len(vills)>num_vills:
            random.shuffle(vills)
            vills=vills[:num_vills]
        for vill in vills:
            mating(vill)
        if len(vills)==0:
            break
        if num%10==0:
            cycle=0
            incest=0
            for vill in vills:
                value_ls=[clan.value for clan in vill.clans]
                woman_ls=[clan.woman_value for clan in vill.clans]
                cur=cluster(value_ls,woman_ls)
                cycle+=cur
                if cur>1:
                    incest+=1
            cycle=cycle/len(vills)
            cycles.append(cycle)
            incest=incest/len(vills)
            incests.append(incest)
        num+=1
    if len(vills)==0:
        cycles=0
    return [cycles,incests]

def run():
    df_cluster=pd.DataFrame(index=cluster_ls)
    df_incest=pd.DataFrame(index=cluster_ls)
    k=0
    for l in range(100):
        res=main()
        if res[0]==0:
            continue
        df_cluster[k]=res[0]
        df_incest[k]=res[1]
        k+=1
    df_cluster.to_csv("./cluster/{}vills_{}lineages_initial{}_coop{}pc_conflict{}pc_marry{}_mutation{}pc_friendship{}_birth{}_epsilon{}pc.csv".format(num_vills,num_lineage,initial,round(coop*100),round(conflict*100),marry,round(mutation*100),friendship,birth,round(epsilon*100)))
    df_incest.to_csv("./incest/{}vills_{}lineages_initial{}_coop{}pc_conflict{}pc_marry{}_mutation{}pc_friendship{}_birth{}_epsilon{}pc.csv".format(num_vills,num_lineage,initial,round(coop*100),round(conflict*100),marry,round(mutation*100),friendship,birth,round(epsilon*100)))

cluster_ls=[]
for i in range(50):
    cluster_ls.append(i*10)

#settings
mutation=0.03
initial_pop=5
num_vills=50
coop=0.1
conflict=3
marry=1
num_lineage=30
friendship=1
birth=4
epsilon=1
initial=1

for mutation in [0.03,0.3]:
    for coop in [0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][2*(int(sys.argv[1])//10):2*(int(sys.argv[1])//10+1)]:
        for conflict in [[0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][int(sys.argv[1])%10]]:
            run()
