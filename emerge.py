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
import x_means_aic

import warnings
warnings.filterwarnings('ignore')

# os.chdir('/Users/kenjiitao 1/Documents/python/clan/clan_emerge')

if int(sys.argv[1])==0:
    if not os.path.exists("./cluster_aic"):
        os.mkdir("./cluster_aic")
    if not os.path.exists("./incest"):
        os.mkdir("./incest")

#memo
# 0105との違いは進化の入るタイミング

def cluster_aic(x,y):
    try:
        cluster=x_means_aic.x_means_cluster(np.c_[x,y])
        centers=cluster.cluster_centers_
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
    clans=vill.clans
    remove_ls=[clan for clan in clans if clan.couple==0]
    for clan in remove_ls:
        clans.remove(clan)
    value_ls=np.array([clan.value for clan in clans])
    for clan in clans:
        dist=np.array([(clan.value-value_ls)**2,(clan.woman_value-value_ls)**2])
        distance=dist.min(axis=0)
        friend=np.sum(np.exp(-distance*friendship))/len(clans)
        rate=1/(1+coop*(1-friend))
        couple=birth*clan.couple
        clan.man=round(np.random.poisson(lam=couple)*rate)
        clan.woman=round(np.random.poisson(lam=couple)*rate)
        clan.couple=0
    remove_ls=[]
    for clan in clans:
        if clan.man+clan.woman>4*initial_pop:
            n=math.floor(math.log2((clan.man+clan.woman)/2/initial_pop))
            clan.man=round(clan.man/2**n)
            clan.woman=round(clan.woman/2**n)
            for i in [0]*(2**n-1):
                clans.append(Clan(clan.value,clan.woman_value,0))
                clans[-1].man=clan.man
                clans[-1].woman=clan.woman
        if clan.man<=0 or clan.woman<=0:
            remove_ls.append(clan)
        else:
            clan.value+=mutation*(2*random.random()-1)
            clan.woman_value+=mutation*(2*random.random()-1)
    for clan in remove_ls:
        clans.remove(clan)
    woman_value_ls=np.array([clan.woman_value for clan in clans])
    for clan in clans:
        enemy=np.sum(np.exp(-(clan.woman_value-woman_value_ls)**2*friendship))/len(clans)
        rate=1/(1+conflict*enemy)
        clan.man=round(clan.man*rate)
        clan.woman=round(clan.woman*rate)
        vill.population+=clan.man+clan.woman

def mating(vill):
    clans=vill.clans
    value_ls=np.array([clan.woman_value for clan in clans])
    for i in range(marry):
        for clan in clans:
            if clan.man>0:
                mates=np.exp(-(clan.value-value_ls)**2*friendship)
                w2 = mates / np.sum(mates)
                mate = np.random.choice(clans, p=w2)
                mate.candidate.append(clan)
        for mate in clans:
            if mate.woman<=0:
                continue
            c=len(mate.candidate)
            if c==0:
                continue
            elif c==1:
                clan=mate.candidate[0]
                couple=min(clan.man,mate.woman)
                clan.man-=couple
                mate.woman-=couple
                clan.couple+=couple
            else:
                id_ls=mate.candidate
                random.shuffle(id_ls)
                for clan in id_ls:
                    couple=min(clan.man,mate.woman)
                    clan.man-=couple
                    mate.woman-=couple
                    clan.couple+=couple
                    if mate.woman<=0:
                        break
            mate.candidate=[]

def main():
    global num

    vills=[]
    num=0
    initial_population=initial_pop*num_lineage*2
    for i in range(num_vills):
        vills.append(Village())
        for j in range(num_lineage):
            vills[i].clans.append(Clan(random.random(),random.random(),initial_pop))
            # vills[i].clans.append(Clan(0,0,initial_pop))
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
                vill.clans=vill.clans[k:]
                vills.append(Village())
                vills[-1].clans=copy.deepcopy(clans)
        while len(vills)>num_vills:
            vills.remove(random.choice(vills))
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
                cur=cluster_aic(value_ls,woman_ls)
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
incests
def run():
    df_cluster_aic=pd.DataFrame(index=cluster_ls)
    df_incest=pd.DataFrame(index=cluster_ls)
    k=0
    for l in range(10):
        res=main()
        if res[0]==0:
            continue
        df_cluster_aic[k]=res[0]
        df_incest[k]=res[1]
        k+=1
    df_cluster_aic.to_csv("./cluster_aic/{}vills_{}lineages_initial{}_coop{}pc_conflict{}pc_marry{}_mutation{}pc_friendship{}.csv".format(num_vills,num_lineage,initial_pop,round(coop*100),round(conflict*100),marry,round(mutation*100),friendship))
    df_incest.to_csv("./incest/{}vills_{}lineages_initial{}_coop{}pc_conflict{}pc_marry{}_mutation{}pc_friendship{}.csv".format(num_vills,num_lineage,initial_pop,round(coop*100),round(conflict*100),marry,round(mutation*100),friendship))

cluster_ls=[]
for i in range(50):
    cluster_ls.append(i*10)
# df_res=vills[growth_ls[0][1]].df_res.T

#settings
mutation=0.01
initial_pop=10
num_vills=30
coop=0.1
conflict=0.1
marry=3
num_lineage=30
friendship=30
for coop in [0.1,0.2,0.3,0.5,1,2,3,5,10,20][2*(int(sys.argv[1])//5):2*(int(sys.argv[1])//5+1)]:
    for conflict in [0.1,0.2,0.3,0.5,1,2,3,5,10,20][2*(int(sys.argv[1])%5):2*(int(sys.argv[1])%5+1)]:
        for mutation in [0.01,0.02,0.03,0.05,0.1,0.2,0.3]:
            birth=2+coop*conflict/2
            if os.path.exists("./cluster_aic/{}vills_{}lineages_initial{}_coop{}pc_conflict{}pc_marry{}_mutation{}pc_friendship{}.csv".format(num_vills,num_lineage,initial_pop,round(coop*100),round(conflict*100),marry,round(mutation*100),friendship)):
                pass
            else:
                run()

# for coop in [[0.3,0.5,0.7,0.9][int(sys.argv[1])//10]]:
#     conflict=coop
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# mutation=0.05
# for conflict in [[0.3,0.5,0.7,0.9][int(sys.argv[1])//10]]:
#     for coop in [[0.1*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# coop=conflict=0.5
# for friendship in [[10,20,40,50][int(sys.argv[1])//10]]:
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# friendship=30
# for num_vills in [[10,20,40,50][int(sys.argv[1])//10]]:
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# num_vills=30
# for num_lineage in [[10,20,40,50][int(sys.argv[1])//10]]:
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# num_lineage=30
# for marry in [[1,2,4,5][int(sys.argv[1])//10]]:
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
# marry=3
# for initial_pop in [[30,20,5,2][int(sys.argv[1])//10]]:
#     for mutation in [[0.01*i for i in range(1,11)][int(sys.argv[1])%10]]:
#         run()
