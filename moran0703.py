import numpy as np
import pandas as pd
import random
import sys
import os
# import csv
import math
import copy
# import collections
import pyclustering
from pyclustering.cluster import xmeans

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import x_means
# os.chdir('/Users/kenjiitao 1/Documents/python/clan/clan_emerge/vill')
# import x_means_aic

import warnings
warnings.filterwarnings('ignore')

if int(sys.argv[1])==0:
    if not os.path.exists("./res"):
        os.mkdir("./res")

#memo
# まずは村単位での粗視化をしないモデル
# L2ノルムで距離を測っている．これは高次元への閣僚を見据えて．
# descentはtraitの出自．０が父系，１が母系
# 今までのモデルに近くて女をよその村からもらったらその村のみんなと仲良くする．

cluster_ls=[]
for i in range(50):
    cluster_ls.append(i*10)

def cluster_aic(data):
    res=0
    structure=0
    data=np.array(data)
    clans=[]
    for descent in [0,1]:
        try:
            init_center = xmeans.kmeans_plusplus_initializer(data[data[:,0]==descent], 2).initialize()
            xm = xmeans.xmeans(data[data[:,0]==descent], init_center, ccore=True)
            xm.process()
            # sizes = [len(cluster) for cluster in xm.get_clusters()]
            # centers=xm.get_centers()
            # for i in range(len(sizes)):
            #     if sizes[i]>num_lineage/10:
            #         clans.append(centers[i])
            clans=xm.get_centers()
        except:
            continue
    if len(clans)>0:
        num_clans=len(clans)
        clan_ls=[]
        for i in range(num_clans):
            mate=0
            child=0
            cur_mate=100
            cur_child=100
            for j in range(num_clans):
                mate_cur=(clans[i][3]-clans[j][1])**2+(clans[i][4]-clans[j][2])**2
                if mate_cur<cur_mate:
                    mate=j
                    cur_mate=mate_cur
            if clans[i][0]==0:
                clan_ls.append([i,mate,[i]])
            else:
                clan_ls.append([i,mate,[]])

        for i in range(len(clan_ls)):
            if clan_ls[i][2]==[]:
                mates=[mate for mate in clan_ls if mate[1]==i]
                children=[]
                for mate in mates:
                    cur_child=100
                    for j in range(num_clans):
                        child_cur=(clans[i][1]-clans[j][1])**2+(clans[mate[0]][2]-clans[j][2])**2
                        if child_cur<cur_child:
                            child=j
                            cur_child=child_cur
                    children.append(child)
                clan_ls[i][2]=children
        counter=1
        for clan in clan_ls:
            if len(clan[2])==1:
                clan[2]=clan[2][0]
            elif len(clan[2])>1:
                counter=counter*len(clan[2])
        clan_ls_ls=[]
        clan_ls_ori=copy.deepcopy(clan_ls[:])
        for i in range(counter):
            clan_ls=copy.deepcopy(clan_ls_ori[:])
            for clan in clan_ls:
                if type(clan[2])!=type(0):
                    if len(clan[2])==0:
                        clan[2]=-1
                    else:
                        clan[2]=clan[2][counter%len(clan[2])]
            clan_ls_ls.append(clan_ls)

        cur_man_cycle=0
        cur_cycle=0
        cur_woman_cycle=0
        num_clans=0
        for clan_ls in clan_ls_ls:
            candidate=list(range(len(clans)))
            while len(candidate)>0:
                marriage_path=[]
                cur=candidate[-1]
                man_path=[cur]
                vill_ls=[]
                while True:
                    next=clan_ls[cur][2]
                    if next in man_path:
                        man_path=man_path[man_path.index(next):]
                        break
                    else:
                        man_path.append(next)
                        cur=next
                cur_woman_cycle_cur=0
                for clan in man_path:
                    if clan not in marriage_path:
                        cur_path=[clan]
                        cur=clan
                        while True:
                            next=clan_ls[clan_ls[cur][1]][2]
                            if next in cur_path:
                                marriage_path.extend(cur_path[cur_path.index(next):])
                                if len(cur_path[cur_path.index(next):])>cur_woman_cycle_cur:
                                    cur_woman_cycle_cur=len(cur_path[cur_path.index(next):])
                                break
                            else:
                                cur_path.append(next)
                                cur=next
                marriage_path=list(set(marriage_path))
                candidate.pop()
                for man in man_path:
                    if man not in marriage_path:
                        man_path.remove(man)
                    if man in candidate:
                        candidate.remove(man)

                if len(marriage_path)>cur_cycle:
                    cur_cycle=len(marriage_path)
                    cur_man_cycle=len(man_path)
                    cur_woman_cycle=cur_woman_cycle_cur
                elif len(marriage_path)==cur_cycle and len(man_path)>cur_man_cycle:
                    cur_cycle=len(marriage_path)
                    cur_man_cycle=len(man_path)
                    cur_woman_cycle=cur_woman_cycle_cur
                elif len(marriage_path)==cur_cycle and len(man_path)==cur_man_cycle and cur_woman_cycle_cur>cur_woman_cycle:
                    cur_cycle=len(marriage_path)
                    cur_man_cycle=len(man_path)
                    cur_woman_cycle=cur_woman_cycle_cur
                else:
                    continue
                rest=0
                for man in marriage_path:
                    if len(clan_ls[man])==len(set(clan_ls[man])):
                        rest+=1
                if rest>=cur_cycle/2:
                    rest=1
                else:
                    rest=0
            clan_ls=np.array(clan_ls)
            ind_ls=[clan_ls[i,2] for i in list(set(clan_ls[:,1]))]
            clan_ls=[list(clan) for clan in clan_ls if clan[0] in ind_ls]
            if len(clan_ls)>num_clans:
                num_clans=len(clan_ls)

        if cur_cycle*cur_man_cycle*cur_woman_cycle!=0:
            if cur_woman_cycle>1 and cur_man_cycle>1:
                structure=4
            elif cur_cycle==1:
                structure=1
            elif cur_cycle<=num_clans/3:
                structure=5
            elif cur_man_cycle*cur_woman_cycle==2:
                structure=2
            elif cur_woman_cycle>2 and cur_man_cycle==1:
                structure=3
            elif cur_woman_cycle==1 and cur_man_cycle>2:
                structure=3
            else:
                structure=6
        # structures=["dead","incest", "dual", "generalized", "restricted", "vill division", "others"]

            res=[cur_cycle,cur_man_cycle,cur_woman_cycle,rest]
    return [structure,res]

class Village:
    def __init__(self):
        self.lineages=[]
        self.population=0

class Lineage:
    def __init__(self,trait,preference,descent,num_couple):
        self.trait=trait
        self.preference=preference
        self.couple=num_couple
        self.man=0
        self.woman=0
        self.child=trait
        self.descent=descent
        self.candidate=[]

def year(vill):
    vill.population=0
    lineages=[lineage for lineage in vill.lineages if lineage.couple>0]
    traits=np.array([lineage.trait for lineage in lineages])
    for lineage in lineages:
        if lineage.descent==0:
            distance=np.array([np.sum((traits-lineage.trait)**2,axis=1),np.sum((traits-lineage.preference)**2,axis=1)])
        else:
            distance=np.array([np.sum((traits-lineage.trait)**2,axis=1),np.sum((traits-lineage.preference)**2,axis=1),np.sum((traits-lineage.child)**2,axis=1)])
        distance=distance.min(axis=0)
        friend=np.sum(np.exp(-distance))/len(lineages)
        rate=1/(1+coop*(1-friend))
        couple=birth*lineage.couple
        lineage.man=round(np.random.poisson(lam=couple)*rate)
        lineage.woman=round(np.random.poisson(lam=couple)*rate)
        lineage.couple=0
    for lineage in lineages:
        if min(lineage.man,lineage.woman)>2*initial_pop:
            n=math.floor(math.log2(min(lineage.man,lineage.woman)/initial_pop))
            lineage.man=round(lineage.man/2**n)
            lineage.woman=round(lineage.woman/2**n)
            for i in [0]*(2**n-1):
                lineages.append(Lineage(lineage.trait,lineage.preference,lineage.descent,0))
                lineages[-1].man=lineage.man
                lineages[-1].woman=lineage.woman
    lineages=[lineage for lineage in lineages if lineage.man*lineage.woman>0]
    for lineage in lineages:
        if random.random()<descent_mut:
            lineage.descent=(lineage.descent+1)%2
        lineage.trait+=np.random.uniform(-mutation,mutation,2)
        lineage.preference+=np.random.uniform(-mutation,mutation,2)
    preference_ls=np.array([lineage.preference for lineage in lineages])
    for lineage in lineages:
        enemy=np.sum(np.exp(-np.sum((preference_ls-lineage.preference)**2,axis=1)))/len(lineages)
        rate=1/(1+conflict*enemy)
        lineage.man=round(lineage.man*rate)
        lineage.woman=round(lineage.woman*rate)
        vill.population+=lineage.man+lineage.woman
    vill.lineages=lineages

def mating(vill):
    lineages=vill.lineages
    for i in range(marry):
        mates=np.array([mate.preference for mate in lineages])
        child_traits=np.array([child.trait for child in lineages if child.descent==1])
        children=np.array([child for child in lineages if child.descent==1])
        for lineage in lineages:
            if lineage.man>0:
                dist=np.exp(-np.sum((mates-lineage.trait)**2,axis=1))
                dist=dist/np.sum(dist)
                mate = np.random.choice(lineages, p=dist)
                mate.candidate.append(lineage)
        for mate in lineages:
            if mate.woman<1 or len(mate.candidate)==0:
                continue
            random.shuffle(mate.candidate)
            for lineage in mate.candidate:
                if mate.woman<1:
                    break
                couple=min(lineage.man,mate.woman)
                lineage.man-=couple
                mate.woman-=couple
                if lineage.descent==1:
                    dist=np.exp(-np.sum((child_traits-np.array([lineage.trait[0],mate.trait[1]]))**2,axis=1))
                    dist=dist/np.sum(dist)
                    child = np.random.choice(children, p=dist)
                    child.couple+=couple
                    lineage.child=child.trait
                else:
                    lineage.couple+=couple
            mate.candidate=[]

def main():
    global num

    num=0
    vills=[]
    cycles=[]
    man_cycles=[]
    woman_cycles=[]
    restricts=[]
    num_clans_ls=[]
    structures=[]
    initial_population=initial_pop*num_lineage*2
    for i in range(num_trial):
        vills.append(Village())
        for j in range(num_lineage):
            # if j % 4==0:
            #     vills[i].lineages.append(Lineage(0,1,0.5,1.5,initial_pop))
            # elif j % 4 ==1:
            #     vills[i].lineages.append(Lineage(0.5,-0.5,0.5,1.5,initial_pop))
            # elif j % 4 ==2:
            #     vills[i].lineages.append(Lineage(1,0,1.5,0.5,initial_pop))
            # elif j % 4 ==3:
            #     vills[i].lineages.append(Lineage(-0.5,0.5,1.5,0.5,initial_pop))
            if initial == 0:
                vills[i].lineages.append(Lineage(np.array([0.0,0.0]),np.array([0.0,0.0]),1,initial_pop))
            elif initial ==1:
                # vills[i].lineages.append(Lineage(np.random.normal(0,friendship/2,2),np.random.normal(0,friendship/2,2),1,initial_pop))
                vills[i].lineages.append(Lineage(np.random.rand(2),np.random.rand(2),1,initial_pop))
            else:
                vills[i].lineages.append(Lineage(5*np.random.rand(2),5*np.random.rand(2),1,initial_pop))


    while num <500:
        # if num == 50:
        #     birth=4
        # counter=0
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
            random.shuffle(vill.lineages)
            n=math.floor(math.log2(vill.population/initial_population))
            k=round(len(vill.lineages)/2**n)
            for i in [0]*(2**n-1):
                lineages=vill.lineages[:k]
                vill.lineages=vill.lineages[k:]
                vills.append(Village())
                vills[-1].lineages=copy.deepcopy(lineages)
                # counter+=1
        while len(vills)>num_trial:
            random.shuffle(vills)
            vills=vills[:num_trial]
        for vill in vills:
            mating(vill)
        if len(vills)==0:
            break
        if num%10==0:
            cur=np.array([0.0]*4)
            n=0
            cur_structures=[0]*7
            for vill in vills:
                data=[[lineage.descent,lineage.trait[0],lineage.trait[1],lineage.preference[0],lineage.preference[1]] for lineage in vill.lineages]
                res=cluster_aic(data)
                if res!=0:
                    n+=1
                    cur+=np.array(res[1])
                    structure=res[0]
                else:
                    structure=0
                cur_structures[structure]+=1
            if n>0:
                cur=cur/n
            cycles.append(round(cur[0],3))
            man_cycles.append(round(cur[1],3))
            woman_cycles.append(round(cur[2],3))
            restricts.append(round(cur[3],3))
            structures.append(cur_structures.index(max(cur_structures)))
            print(cur_structures)
        # print(counter)
        num+=1
    if len(vills)==0:
        cycles=0
    return [cycles,man_cycles,woman_cycles,restricts,structures]
structures
cur_structures
def run():
    df=pd.DataFrame(index=list(range(50)))
    k=0
    for l in range(10):
        try:
            res=main()
            if res[0]==0:
                continue
            else:
                df[k]=np.array(res).T.tolist()
                k+=1
        except:
            pass
    df.to_pickle("./res/{}regions_{}lineages_coop{}pc_conflict{}pc_mutation{}pm_descentmut{}pm_marry{}_friendship{}_initial{}_birth{}.pkl".format(num_trial,num_lineage,round(coop*100),round(conflict*100),round(mutation*1000),round(descent_mut*1000),marry,friendship,initial,birth))

#settings
initial=2
num_lineage=60
initial_pop=10
num_trial=100
friendship=1
descent_mut=0.001
marry=3
mutation=0.03
coop=0.5
conflict=5
initial=1
birth=4

for descent_mut in [0.0001,0.001,0.01]:
    for mutation in [0.0001,0.001,0.01,0.1][2*(int(sys.argv[2])):2*(int(sys.argv[2])+1)]:
        for coop in [[0.05,0.1,0.3,0.5,1][int(sys.argv[1])//5]]:
            for conflict in [[0.5,1,2,3,5][int(sys.argv[1])%5]]:
                run()
# birth=6
# num_vills=3
# for coop in [0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][2*(int(sys.argv[1])//10):2*(int(sys.argv[1])//10+1)]:
#     for conflict in [[0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][int(sys.argv[1])%10]]:
#         run()
# else:
#     num_vills=2
#     coop=0.5
#     conflict=2
#     birth=2+coop*conflict/num_vills
#     if int(sys.argv[1])<7:
#         for mutation in [[0.01,0.02,0.03,0.05,0.1,0.2,0.3][int(sys.argv[1])%7]]:
#             run()
#     elif int(sys.argv[1])<14:
#         for descent_mut in [[0.001,0.002,0.003,0.005,0.01,0.02,0.03][(int(sys.argv[1])-7)%7]]:
#             run()
#     else:
#         for num_lineage in [[10,20,30,40][(int(sys.argv[1])-14)//4]]:
#             for num_trial in [[10,20,30,40][(int(sys.argv[1])-14)%4]]:
#                 run()
