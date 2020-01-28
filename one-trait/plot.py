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
# import x_means
from pyclustering.cluster import xmeans
import seaborn as sns
sns.set_style(style="whitegrid")
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

if int(sys.argv[1])==0:
    if not os.path.exists("./figs_timeseries"):
        os.mkdir("./figs_timeseries")
    if not os.path.exists("./figs_map"):
        os.mkdir("./figs_map")

class Village:
    def __init__(self):
        self.clans=[]
        self.population=0
        self.df=pd.DataFrame()

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

    return [[clan.value for clan in vill.clans],[clan.woman_value for clan in vill.clans]]


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
            vills[i].clans.append(Clan(0,0,initial_pop))

    cycles=[]
    incests=[]
    while num <500:
        if num ==20:
            mutation=0.1
        remove_ls=[]
        duplicate_ls=[]
        for vill in vills:
            vill.df[num]=year(vill)
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
                vills[-1].df=vill.df.copy()
        if len(vills)>num_vills:
            random.shuffle(vills)
            vills=vills[:num_vills]
        for vill in vills:
            mating(vill)
        if len(vills)==0:
            break
        num+=1
    if len(vills)==0:
        cycles=0
    for k in range(min(50,len(vills))):
        vill=vills[k]
        my_ls=[]
        for i in range(500):
            my_ls.extend([[i,vill.df.iat[0,i][j],vill.df.iat[1,i][j]] for j in range(len(vill.df.iat[0,i]))])
        df_res=pd.DataFrame(my_ls,columns=["time","t","p"])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(df_res["time"],df_res["t"],s=0.5,color="blue")
        ax.scatter(df_res["time"],df_res["p"],s=0.5,color="red")
        ax.set_xlabel("time",fontsize=36)
        ax.tick_params(labelsize=24)
        fig.tight_layout()
        fig.savefig("figs_timeseries/timeseries_mutation{}pc_coop{}pc_conf{}pc_{}_{}.eps".format(round(mutation*100),round(coop*100),round(conflict*100),k,trial))
        plt.close('all')

        data = np.array([vill.df.iat[0,-1],vill.df.iat[1,-1]]).T
        init_center = xmeans.kmeans_plusplus_initializer(data, 1).initialize()
        xm = xmeans.xmeans(data, init_center, ccore=False)
        xm.process()
        sizes = [len(cluster) for cluster in xm.get_clusters()]
        centers=xm.get_centers()
        clusters_candidate=xm.get_clusters()
        ls=[]
        for i in range(len(sizes)):
            if sizes[i]>num_lineage/10:
                ls.append(clusters_candidate[i])
        # clans=xm.get_centers()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i in range(len(ls)):
            try:
                ax.scatter(data[:,0][ls[i]],data[:,1][ls[i]],s=60,c=current_palette[i])
            except:
                pass
        ax.set_xlabel(r"$t$",fontsize=36)
        ax.set_ylabel(r"$p$",fontsize=36)
        ax.tick_params(labelsize=24)
        ax.set_aspect('equal', 'datalim')
        fig.tight_layout()
        fig.savefig("figs_map/map_mutation{}pc_coop{}pc_conf{}pc_{}_{}.eps".format(round(mutation*100),round(coop*100),round(conflict*100),k,trial))


        # x_means.x_means_plt(vill.df.iat[0,-1],vill.df.iat[1,-1],"figs_map/map_mutation{}pc_coop{}pc_conf{}pc_{}_{}.eps".format(round(mutation*100),round(coop*100),round(conflict*100),k,trial))
#settings
mutation=0.3
initial_pop=5
num_vills=50
coop=0.1
conflict=0.1
marry=1
num_lineage=30
friendship=1
birth=4
epsilon=1
initial=1
current_palette = sns.color_palette("Set1", 4)

for trial in range(5):
    if int(sys.argv[1])==0:
        coop=0.1
        conflict=0.1
        main()
    elif int(sys.argv[1])==1:
        coop=0.5
        conflict=0.5
        main()
    elif int(sys.argv[1])==2:
        coop=0.5
        conflict=1.0
        main()
    elif int(sys.argv[1])==3:
        coop=0.5
        conflict=2.0
        main()
