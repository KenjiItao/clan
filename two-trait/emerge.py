import numpy as np
import pandas as pd
import random
import sys
import os
import math
import copy
import pyclustering
from pyclustering.cluster import xmeans


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
def cluster(data):
    res=0
    structure=0
    data=np.array(data)
    clans=[]
    for descent in [0,1]:
        try:
            init_center = xmeans.kmeans_plusplus_initializer(data[data[:,0]==descent], 2).initialize()
            xm = xmeans.xmeans(data[data[:,0]==descent], init_center, ccore=False)
            xm.process()
            sizes = [len(cluster) for cluster in xm.get_clusters()]
            centers=xm.get_centers()
            for i in range(len(sizes)):
                if sizes[i]>num_lineage/10:
                    clans.append(centers[i])
            # clans=xm.get_centers()
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
                        if clans[j][0]==1:
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
                    elif next == -1:
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
                            elif next == -1:
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
            if cur_woman_cycle>1 and cur_man_cycle>1 and cur_cycle>3:
                structure=4
            elif cur_cycle==1:
                structure=1
            elif cur_cycle<=num_clans/3:
                structure=5
            elif cur_cycle==2:
                structure=2
            elif cur_woman_cycle>2 or cur_man_cycle>2:
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
        self.father=trait
        self.descent=descent
        self.candidate=[]

def year(vill):
    vill.population=0
    lineages=[lineage for lineage in vill.lineages if lineage.couple>0]
    traits=np.array([lineage.trait for lineage in lineages])
    fathers=np.array([lineage.father for lineage in lineages])
    preferences=np.array([lineage.preference for lineage in lineages])
    for lineage in lineages:
        distance=np.array([np.sum((traits-lineage.trait)**2,axis=1),np.sum((traits-lineage.preference)**2,axis=1),np.sum((preferences-lineage.trait)**2,axis=1),np.sum((fathers-lineage.trait)**2,axis=1),np.sum((traits-lineage.father)**2,axis=1)]).min(axis=0)
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
                lineages.append(Lineage(np.copy(lineage.trait),np.copy(lineage.preference),lineage.descent,0))
                lineages[-1].man=lineage.man
                lineages[-1].woman=lineage.woman
    lineages=[lineage for lineage in lineages if lineage.man*lineage.woman>0]
    for lineage in lineages:
        if random.random()<descent_mut:
            lineage.descent=(lineage.descent+1)%2
        lineage.trait+=np.random.uniform(-mutation,mutation,2)
        lineage.preference+=np.random.uniform(-mutation,mutation,2)
    preferences=np.array([lineage.preference for lineage in lineages])
    for lineage in lineages:
        enemy=np.sum(np.exp(-np.sum((preferences-lineage.preference)**2,axis=1)))/len(lineages)
        rate=1/(1+conflict*enemy)
        lineage.man=round(lineage.man*rate)
        lineage.woman=round(lineage.woman*rate)
        vill.population+=lineage.man+lineage.woman
    vill.lineages=lineages[:]

def mating(vill):
    lineages=vill.lineages
    mates=np.array([mate.preference for mate in lineages])
    for lineage in lineages:
        if lineage.man>0:
            dist=np.exp(-np.sum((mates-lineage.trait)**2,axis=1))
            dist=dist/np.sum(dist)
            mate = np.random.choice(lineages, p=dist)
            mate.candidate.append(lineage)
    for mate in lineages:
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
            if lineage.descent==1:
                lineages.append(Lineage(np.array([lineage.trait[0],mate.trait[1]]),np.array([lineage.preference[0],mate.preference[1]]),lineage.descent,couple))
                lineages[-1].father=np.copy(lineage.trait)
            else:
                lineage.couple+=couple
                lineage.father=np.copy(lineage.trait)
        mate.candidate=[]

def main():
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
             vills[i].lineages.append(Lineage(np.random.normal(0,friendship,2),np.random.normal(0,friendship,2),1*(random.random()<0.5),initial_pop))

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
            random.shuffle(vill.lineages)
            n=math.floor(math.log2(vill.population/initial_population))
            k=round(len(vill.lineages)/2**n)
            for i in [0]*(2**n-1):
                lineages=vill.lineages[:k]
                vill.lineages=vill.lineages[k:]
                vills.append(Village())
                vills[-1].lineages=copy.deepcopy(lineages)
        if len(vills)>num_trial:
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
                res=cluster(data)
                if res!=0:
                    n+=1
                    cur+=np.array(res[1])
                    structure=res[0]
                else:
                    structure=0
                cur_structures[structure]+=1
            if n>0:
                cur=cur/n
            cur_structures=cur_structures[:-1]
            cycles.append(round(cur[0],3))
            man_cycles.append(round(cur[1],3))
            woman_cycles.append(round(cur[2],3))
            restricts.append(round(cur[3],3))
            structures.append(cur_structures.index(max(cur_structures)))
        num+=1
    if len(vills)==0:
        cycles=0
    return [cycles,man_cycles,woman_cycles,restricts,structures]
num
def run():
    df=pd.DataFrame(index=list(range(50)))
    k=0
    for l in range(50):
        try:
            res=main()
            if res[0]==0:
                continue
            else:
                df[k]=np.array(res).T.tolist()
                k+=1
        except:
            pass
    df.to_pickle("./res/{}regions_{}lineages_coop{}pc_conflict{}pc_mutation{}pm_descentmut{}pm_marry{}_friendship{}_initial{}_birth{}_epsilon{}pc.pkl".format(num_trial,num_lineage,round(coop*100),round(conflict*100),round(mutation*1000),round(descent_mut*1000),marry,friendship,initial,birth,round(epsilon*100)))

#settings

num_lineage=50
initial_pop=5
num_trial=100
friendship=1
descent_mut=0.01
marry=1
mutation=0.5
coop=3
conflict=0.3
initial=1
birth=4
epsilon=1

# for mutation in [[0.03,0.1][int(sys.argv[2])]]:
#     for coop in [0.05,0.1,0.2,0.3,0.5,1,2,3,5][3*(int(sys.argv[1])//10):3*(int(sys.argv[1])//10+1)]:
#         for conflict in [[0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][int(sys.argv[1])%10]]:
#             run()

for coop in [0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][2*(int(sys.argv[1])//10):2*(int(sys.argv[1])//10+1)]:
    for conflict in [[0.05,0.1,0.2,0.3,0.5,1,2,3,5,10][int(sys.argv[1])%10]]:
        run()
