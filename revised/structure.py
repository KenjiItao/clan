import numpy as np
import pandas as pd
import random
import sys
import os
import math
import copy
import pyclustering
from pyclustering.cluster import xmeans

import seaborn as sns
from matplotlib import pyplot as plt

plt.switch_backend('agg')
sns.set(style='whitegrid')
current_palette = sns.color_palette("colorblind", 5)
if True:
    current_palette[0] = (0 / 255, 114 / 255, 178 / 255)
    current_palette[1] = (240 / 255, 228 / 255, 66 / 255)
    current_palette[2] = (0 / 255, 158 / 255, 115 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)


import warnings
warnings.filterwarnings('ignore')

if int(sys.argv[1])==0:
    if not os.path.exists("./res"):
        os.mkdir("./res")
        os.mkdir("./res2")
        os.mkdir("./figs")

class Society:
    def __init__(self):
        self.families = []
        self.df=pd.DataFrame()

class Family:
    def __init__(self,trait,preference, counter):
        self.trait = trait
        self.preference = preference
        self.man = 0
        self.woman = 0
        self.counter = counter


def my_distance(x, y):
    # return np.sum(((x - y) / sigma)**2,axis=1)
    return np.sum((x - y)**2,axis=1)


def generation(families):
    traits  = np.array([family.trait for family in families])
    preferences = np.array([family.preference for family in families])
    for family in families:
        if family.counter == chance:
            distance = np.array([my_distance(traits, family.trait),my_distance(traits, family.preference), my_distance(preferences, family.trait)]).min(axis=0)
            friend = np.sum(np.exp(- distance)) / len(families)
            rival = np.sum(np.exp(- my_distance(preferences, family.preference))) / len(families)
            rate = math.exp(- d_c * (1 - friend) - d_m * rival)
            couple = birth * rate
            family.man = np.random.poisson(lam = couple)
            family.woman = np.random.poisson(lam = couple)
        family.counter -= 1

    families = [family for family in families if family.man * family.woman > 0]

    return families


def mating(families):
    next_generation = []
    random.shuffle(families)
    mates = np.array([mate.preference for mate in families])
    for family in families:
        if family.man > 0:
            dist = np.exp(- my_distance(mates, family.trait))
            dist = dist / np.sum(dist)
            mate = np.random.choice(families, p = dist)

            if mate.woman > 0:
                couple = min(family.man, mate.woman)
                family.man -= couple
                mate.woman -= couple
                for i in range(couple):
                    new_trait = np.array([family.trait[0], mate.trait[1]]) + np.random.normal(0, mutation, 2)
                    new_preference = np.array([family.preference[0], mate.preference[1]]) + np.random.normal(0, mutation, 2)
                    next_generation.append(Family(new_trait, new_preference, chance))

    for family in families:
        if family.counter * family.man * family.woman > 0:
            next_generation.append(family)

    return next_generation

def cluster(data):
    structure = 0
    descent = 0
    data = np.array(data)
    clans = []
    clusters_clan = []
    try:
        init_center = xmeans.kmeans_plusplus_initializer(data, 2).initialize()
        xm = xmeans.xmeans(data, init_center, ccore=False)
        xm.process()
        sizes = [len(cluster) for cluster in xm.get_clusters()]
        centers=xm.get_centers()
        clusters = xm.get_clusters()
        for i in range(len(sizes)):
            if sizes[i] > 5:
                clans.append(centers[i])
                clusters_clan.append(clusters[i])
    except:
        pass
    if len(clans)>0:
        while len(clans) > 0:
            num_clans=len(clans)
            clan_ls=[]
            for i in range(num_clans):
                mate = i
                child = i
                cur_mate = (clans[i][2]-clans[i][0])**2 + (clans[i][3]-clans[i][1])**2 - 1
                cur_child= (clans[i][0]-clans[i][0])**2 + (clans[i][3]-clans[i][1])**2 - 1
                for j in range(num_clans):
                    mate_cur=(clans[i][2]-clans[j][0])**2 + (clans[i][3]-clans[j][1])**2
                    child_cur=(clans[i][0]-clans[j][0])**2 + (clans[i][3]-clans[j][1])**2
                    if mate_cur<cur_mate:
                        mate=j
                        cur_mate=mate_cur
                    if child_cur<cur_child:
                        child=j
                        cur_child=child_cur

                clan_ls.append([i, mate, child])

            cur_ls = list(set(np.array(clan_ls)[:,-1]))
            if len(cur_ls) == num_clans:
                break
            else:
                clans = [clans[i] for i in cur_ls]


        cur_descent_cycle  = 0
        cur_marriage_cycle = 0
        cur_population = 0
        cur_clans = []

        candidate=list(range(len(clans)))
        while len(candidate)>0:
            marriage_path=[]
            cur = candidate[-1]
            man_path = [cur]
            kinship_clans = []
            while True:
                next = clan_ls[cur][2]
                if next in man_path:
                    man_path = man_path[man_path.index(next):]
                    break
                else:
                    man_path.append(next)
                    cur=next
            kinship_clans.extend(man_path)
            descent_cycle = len(man_path)
            cur_woman_cycle_cur=0
            for clan in man_path:
                cur_path = [clan]
                cur = clan
                while True:
                    next = clan_ls[cur][1]
                    if next in cur_path:
                        cur_path = cur_path[cur_path.index(next):]
                        kinship_clans.extend(cur_path)
                        if len(cur_path) > cur_woman_cycle_cur:
                            cur_woman_cycle_cur = len(cur_path)
                        break
                    else:
                        cur_path.append(next)
                        cur = next
            marriage_cycle = cur_woman_cycle_cur
            candidate.pop()
            for man in man_path:
                if man in candidate:
                    candidate.remove(man)
            kinship_clans = list(set(kinship_clans))

            if descent_cycle >= cur_descent_cycle and marriage_cycle >= cur_marriage_cycle and len(kinship_clans) >= len(cur_clans):
                cur_descent_cycle = descent_cycle
                cur_marriage_cycle = marriage_cycle
                cur_clans = kinship_clans[:]

        cur_paternal_cycle  = 0
        clans_ori = clans[:]
        clans = []

        for clan in clans_ori:
            clans.append([clan[0], clan[2]])

        if len(clans)>0:
            num_clans=len(clans)
            clan_ls=[]
            for i in range(num_clans):
                mate = i
                cur_mate = (clans[i][1]-clans[i][0])**2 - 1
                for j in range(num_clans):
                    mate_cur = (clans[i][1]-clans[j][0])**2
                    if mate_cur<cur_mate:
                        mate=j
                        cur_mate=mate_cur

                clan_ls.append([i, mate])

            candidate=list(range(len(clans)))
            while len(candidate)>0:
                cur = candidate[-1]
                marriage_path = [cur]
                kinship_clans = []
                population = 0
                while True:
                    next = clan_ls[cur][1]
                    if next in marriage_path:
                        marriage_path = marriage_path[marriage_path.index(next):]
                        break
                    else:
                        marriage_path.append(next)
                        cur=next

                candidate.pop()
                for clan in marriage_path:
                    if clan in candidate:
                        candidate.remove(clan)

                if len(marriage_path) >= cur_paternal_cycle:
                    cur_paternal_cycle = len(marriage_path)

        clans = []
        cur_maternal_cycle  = 0
        for clan in clans_ori:
            clans.append([clan[1], clan[3]])

        if len(clans)>0:
            num_clans=len(clans)
            clan_ls=[]
            for i in range(num_clans):
                mate = i
                cur_mate = (clans[i][1]-clans[i][0])**2 - 1
                for j in range(num_clans):
                    mate_cur=(clans[i][1]-clans[j][0])**2
                    if mate_cur<cur_mate:
                        mate=j
                        cur_mate=mate_cur

                clan_ls.append([i, mate])

            candidate=list(range(len(clans)))
            while len(candidate)>0:
                cur = candidate[-1]
                marriage_path = [cur]
                kinship_clans = []
                population = 0
                while True:
                    next = clan_ls[cur][1]
                    if next in marriage_path:
                        marriage_path = marriage_path[marriage_path.index(next):]
                        break
                    else:
                        marriage_path.append(next)
                        cur=next

                candidate.pop()
                for clan in marriage_path:
                    if clan in candidate:
                        candidate.remove(clan)

                if len(marriage_path) >= cur_maternal_cycle:
                    cur_maternal_cycle = len(marriage_path)


        if cur_marriage_cycle * cur_descent_cycle == 0:
            structure = 0
        elif cur_marriage_cycle == 1 and cur_descent_cycle == 1:
            structure = 1
        elif cur_marriage_cycle == 2 and cur_descent_cycle == 1:
            structure = 2
        elif cur_marriage_cycle == cur_descent_cycle == len(cur_clans) == 2:
            structure = 2
        elif cur_marriage_cycle > 2 and cur_descent_cycle == 1:
            structure = 3
        elif cur_marriage_cycle == cur_descent_cycle == len(cur_clans):
            structure = 3
        elif cur_marriage_cycle > 1 and cur_descent_cycle > 1 and len(cur_clans) > 3:
            structure = 4
        else:
            structure = 5

        descent = 2 * (cur_paternal_cycle > 1) + 1 * (cur_maternal_cycle > 1)

    return [structure, descent, clusters_clan]

def main(l):
    num = 0
    societies = []
    structures = []
    descents = np.array([[0] * 4] * 4)
    for i in range(num_society):
        societies.append(Society())
        for j in range(num_family):
            societies[i].families.append(Family(np.array([0.0, 0.0]), np.array([0.0, 0.0]), chance))
            # societies[i].families.append(Family(np.random.normal(0, 1, 2), np.random.normal(0, 1, 2), chance))

    while num < iter:
        remove_ls = []
        duplicate_ls = []
        for society in societies:
            society.families = generation(society.families)
            society.df[num] = [[family.trait[0] for family in society.families], [family.trait[1] for family in society.families], [family.preference[0] for family in society.families], [family.preference[1] for family in society.families]]
            society.families = mating(society.families)
            population = len(society.families)
            if population < num_family / 10:
                remove_ls.append(society)
            elif population > num_family * 2:
                duplicate_ls.append(society)
        for society in remove_ls:
            societies.remove(society)
        for society in duplicate_ls:
            population = len(society.families)
            random.shuffle(society.families)
            n = math.floor(math.log2(population / num_family))
            k = round(len(society.families) / 2**n)
            for i in [0] * (2**n - 1):
                families = society.families[:k]
                society.families = society.families[k:]
                societies.append(Society())
                societies[-1].families = copy.deepcopy(families)
                societies[-1].df = society.df.copy()
        if len(societies) > num_society:
            random.shuffle(societies)
            societies = societies[:num_society]
        if len(societies) == 0:
            break

        if num % 10 == 0:
            cur_structures = [0] * 7
            for society in societies:
                data = [[family.trait[0], family.trait[1], family.preference[0], family.preference[1]] for family in society.families]
                [structure, descent] = cluster2(data)[:2]
                cur_structures[structure] += 1
                if structure in [1, 2, 3, 4] and num > 0.6 * iter:
                    descents[structure - 1, descent] += 1
            cur_structures = cur_structures[:-1]
            structures.append(cur_structures.index(max(cur_structures)))

        num += 1
    if len(societies) == 0:
        structures = []


    if num == iter:
        k = 0
        for society in societies:
            flag = 0
            data = [[family.trait[0],family.trait[1],family.preference[0],family.preference[1]] for family in society.families]
            res = cluster2(data)
            data = np.array(data)

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

            my_ls=[]
            for i in range(iter):
                my_ls.extend([[i,society.df.iat[0,i][j],society.df.iat[2,i][j]] for j in range(len(society.df.iat[0,i]))])
            df_res=pd.DataFrame(my_ls,columns=["time","t","p"])
            ax1.scatter(df_res["time"],df_res["t"], s=0.2, color= "blue")
            ax1.scatter(df_res["time"],df_res["p"], s=0.2, color= "red")
            ax1.set_xlabel("generation",fontsize=24)
            ax1.set_ylabel(r"$t_1, p_1$",fontsize=24)
            ax1.tick_params(labelsize = 16)

            my_ls=[]
            for i in range(iter):
                my_ls.extend([[i,society.df.iat[1,i][j],society.df.iat[3,i][j]] for j in range(len(society.df.iat[0,i]))])
            df_res=pd.DataFrame(my_ls,columns=["time","t","p"])
            ax2.scatter(df_res["time"],df_res["t"], s=0.2, color= "blue")
            ax2.scatter(df_res["time"],df_res["p"], s=0.2, color= "red")
            ax2.set_xlabel("generation",fontsize=24)
            ax2.set_ylabel(r"$t_2, p_2$",fontsize=24)
            ax2.tick_params(labelsize = 16)

            fig.tight_layout()
            fig.savefig(f"figs/timeseries_{res[0]}_{num_society}societies_{num_family}families_dc{round(d_c * 100)}_dm{round(d_m * 100)}_mutation{round(mutation * 1000)}pm_ birth{birth}_{l}_{k}.pdf", bbox_inches='tight')
            plt.close('all')

            fig, (axL, axC, axR) = plt.subplots(ncols=3, figsize=(12,4))

            for i in range(len(clusters)):
                try:
                    axL.scatter(data[:, 0][clusters[i]], data[:, 2][clusters[i]], s=100-20*i, c=current_palette[i + 1])
                except:
                    pass
            # axL.scatter(data[:,0], data[:,2], s=80)
            axL.set_xlabel(r"$t_1$",fontsize=24)
            axL.set_ylabel(r"$p_1$",fontsize=24)
            axL.tick_params(labelsize = 16)
            axL.set_aspect('equal', 'datalim')

            for i in range(len(clusters)):
                try:
                    axC.scatter(data[:, 1][clusters[i]], data[:, 3][clusters[i]], s=100-20*i, c=current_palette[i + 1])
                except:
                    pass
            # axC.scatter(data[:,1], data[:,3], s=80)
            axC.set_xlabel(r"$t_2$",fontsize=24)
            axC.set_ylabel(r"$p_2$",fontsize=24)
            axC.tick_params(labelsize = 16)
            axC.set_aspect('equal', 'datalim')

            for i in range(len(clusters)):
                try:
                    axR.scatter(data[:, 0][clusters[i]], data[:, 1][clusters[i]], s=100-20*i, c = current_palette[i + 1])
                except:
                    pass
            # axR.scatter(data[:,0], data[:,1], s=80)
            axR.set_xlabel(r"$t_1$",fontsize=24)
            axR.set_ylabel(r"$t_2$",fontsize=24)
            axR.tick_params(labelsize = 16)
            axR.set_aspect('equal', 'datalim')

            fig.tight_layout()
            fig.savefig(f"figs/structure_{res[0]}_{num_society}societies_{num_family}families_dc{round(d_c * 100)}_dm{round(d_m * 100)}_mutation{round(mutation * 1000)}pm_ birth{birth}_{l}_{k}.pdf", bbox_inches='tight')
            plt.close('all')
            k += 1

    # return [cycles,man_cycles,woman_cycles,restricts,structures]
    return [structures, descents]



def run():
    df = pd.DataFrame(index=list(range(iter // 10)))
    df_descent = pd.DataFrame(index=list(range(16)))
    k = 0
    for l in range(50):
        try:
            res = main(l)
            if len(res[0]) == 0:
                continue
            else:
                df[k] = np.array(res[0]).T.tolist()
                df_descent[k] = np.ravel(res[1])
                k += 1
        except:
            pass
    df.to_csv(f"res/{num_society}societies_{num_family}families_dc{round(d_c * 100)}_dm{round(d_m * 100)}_mutation{round(mutation * 1000)}pm_birth{birth}_sigma{sigma}.csv")
    df_descent.to_csv(f"res2/{num_society}societies_{num_family}families_dc{round(d_c * 100)}_dm{round(d_m * 100)}_mutation{round(mutation * 1000)}pm_birth{birth}.csv")


#settings
num_family = 50
num_society = 50
mutation = 0.1
d_c = 0.2
d_m = 3.0
birth = 5
sigma = 1
iter = 1000
chance = 2

run()
