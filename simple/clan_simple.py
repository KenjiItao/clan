import numpy as np
import pandas as pd
import random
import sys
import os
import math
import copy
from pyclustering.cluster import xmeans

import seaborn as sns
from matplotlib import pyplot as plt

plt.switch_backend('agg')
sns.set(style='whitegrid')
current_palette = sns.color_palette("colorblind", 8)
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

def bayesian_information_criterion(data, clusters, centers):
         """!
         @brief Calculates splitting criterion for input clusters using bayesian information criterion.

         @param[in] clusters (list): Clusters for which splitting criterion should be calculated.
         @param[in] centers (list): Centers of the clusters.

         @return (double) Splitting criterion in line with bayesian information criterion.
                 High value of splitting criterion means that current structure is much better.

         @see __minimum_noiseless_description_length(clusters, centers)

         """

         scores = [float('inf')] * len(clusters)     # splitting criterion
         dimension = len(data[0])

         # estimation of the noise variance in the data set
         sigma_sqrt = 0.0
         K = len(clusters)
         N = 0.0

         for index_cluster in range(0, len(clusters), 1):
             for index_object in clusters[index_cluster]:
                 sigma_sqrt += np.sum(np.square(data[index_object] - centers[index_cluster]))

             N += len(clusters[index_cluster])

         if N - K > 0:
             sigma_sqrt /= (N - K)
             p = (K - 1) + dimension * K + 1

             # in case of the same points, sigma_sqrt can be zero (issue: #407)
             sigma_multiplier = 0.0
             if sigma_sqrt <= 0.0:
                 sigma_multiplier = float('-inf')
             else:
                 sigma_multiplier = dimension * 0.5 * math.log(sigma_sqrt)

             # splitting criterion
             for index_cluster in range(0, len(clusters), 1):
                 n = len(clusters[index_cluster])

                 L = n * math.log(n) - n * math.log(N) - n * 0.5 * math.log(2.0 * np.pi) - n * sigma_multiplier - (n - K) * 0.5

                 # BIC calculation
                 scores[index_cluster] = L - p * 0.5 * math.log(N)

         return sum(scores)

class Society:
    def __init__(self):
        self.families = []
        self.df=pd.DataFrame()

class Family:
    def __init__(self,trait,preference):
        self.trait = trait
        self.preference = preference

def my_distance(x, y):
    # return np.sum(np.exp(-np.abs(x - y)))
    return np.sum(np.exp( -(x - y)**2))

def generation(families, cur_rate):
    traits  = np.array([family.trait for family in families])
    preferences = np.array([family.preference for family in families])
    next_generation = []
    tot_rate = 0
    for family in families:
        # distance = np.array([my_distance(traits, family.trait),my_distance(traits, family.preference), my_distance(preferences, family.trait)]).min(axis=0)
        # friend = np.sum(np.exp(- distance)) / len(families)
        kins = my_distance(traits, family.trait) / len(families)
        # preferring = my_distance(traits, family.preference) / len(families)
        preferring = np.sum(np.exp(- (traits - family.preference) ** 2) * (1 - np.exp(- (traits - family.trait) ** 2))) / len(families)
        rival = my_distance(preferences, family.preference) / len(families)
        preferred = my_distance(preferences, family.trait) / len(families) / kins
        rate = (kins + preferring + r * (1 - rival)) * preferred
        tot_rate += rate
        num_children = np.random.poisson(lam = rate / cur_rate)
        for i in range(num_children):
            next_generation.append(Family(family.trait + random.gauss(0, mutation), family.preference + random.gauss(0, mutation)))

    return next_generation, tot_rate

def cluster(x,y):
    try:
        data = np.c_[x, y]
        init_center = xmeans.kmeans_plusplus_initializer(data, 1).initialize()
        xm1 = xmeans.xmeans(data, init_center, ccore=False)
        xm1.process()
        clusters1= xm1.get_clusters()
        centers1 = xm1.get_centers()

        init_center = xmeans.kmeans_plusplus_initializer(data, 2).initialize()
        xm2 = xmeans.xmeans(data, init_center, ccore=False)
        xm2.process()
        clusters2= xm2.get_clusters()
        centers2 = xm2.get_centers()

        if len(centers1) == len(centers2):
            clusters = clusters2
            clans = centers2
        else:
            if bayesian_information_criterion(data, clusters1, centers1) > bayesian_information_criterion(data, clusters2, centers2):
                clusters = clusters1
                clans = centers1
            else:
                clusters = clusters2
                clans = centers2

        while True:
            merge_ls = []
            remove_ls = []
            for i in range(len(clans)):
                for j in range(i+1, len(clans)):
                    # |t_i - t_j| > 0.83 is the condition for marriage possibility to double.
                    # print(clans[i], clans[j], sum((np.array(clans[i]) - np.array(clans[j]))**2), abs(clans[i][0] - clans[j][0]))
                    if sum((np.array(clans[i]) - np.array(clans[j]))**2) < 1:
                        merge_ls.append([i,j])

            if len(merge_ls) == 0:
                break
            else:
                for merge in merge_ls:
                    if merge[1] not in remove_ls:
                        remove_ls.append(merge[1])
                        clusters[merge[0]] += clusters[merge[1]]
                        # clusters.remove(clusters[merge[1]])
                clans_id = list(set(range(len(clans))) - set(remove_ls))
                clans = [[np.mean(data[clusters[i]][:,0]), np.mean(data[clusters[i]][:,1])] for i in clans_id]
                clusters = [clusters[i] for i in clans_id]

        while len(clans) > 0:
            num_clans=len(clans)
            clan_ls=[]
            for i in range(num_clans):
                mate = i
                cur = (clans[i][1]-clans[i][0])**2
                for j in range(num_clans):
                    if abs(clans[i][1]-clans[j][0])<cur:
                        mate=j
                        cur=abs(clans[i][1]-clans[j][0])

                clan_ls.append([i, mate])

            cur_ls = list(set(np.array(clan_ls)[:,-1]))
            if len(cur_ls) == num_clans:
                break
            else:
                clans = [clans[i] for i in cur_ls]

        candidate=list(range(len(clans)))
        clans = clan_ls[:]

        cur_cycle=0
        counter = 0
        while len(candidate)>0:
            cycle=[candidate[0]]
            cur=candidate[0]
            while True:
                next=clans[cur][1]
                if clans[cur][1] in cycle:
                    if len(cycle)-cycle.index(next) > cur_cycle:
                        cur_cycle = len(cycle)-cycle.index(next)
                    for clan in cycle:
                        if clan in candidate:
                            candidate.remove(clan)
                    counter += 1
                    break
                else:
                    cycle.append(next)
                    cur=next
    except:
        cur_cycle = 1
        counter = 1
        clusters, clans = list(range(len(x))), [[0, 0]]
    return cur_cycle, clusters, len(clans), counter

def main(l):
    num = 0
    societies = []
    cycle_ls = []
    num_clan_ls = []
    num_structure_ls = []
    for i in range(num_society):
        societies.append(Society())
        for j in range(num_family):
            societies[i].families.append(Family(0.0, 0.0))
            # societies[i].families.append(Family(random.gauss(0, 1), random.gauss(0, 1)))
            # societies[i].families.append(Family(random.random(), random.random()))
            # societies[i].families.append(Family(np.random.normal(0, 1, 2), np.random.normal(0, 1, 2), chance))
    tot_pop = num_society * num_family
    tot_rate = num_society * num_family
    while num < iter:
        remove_ls = []
        duplicate_ls = []
        cur_rate = tot_rate / tot_pop
        tot_rate = 0
        for society in societies:
            society.families, rate = generation(society.families, cur_rate)
            tot_rate += rate
            society.df[num] = [[family.trait for family in society.families], [family.preference for family in society.families]]
            population = len(society.families)
            if population == 0:
                remove_ls.append(society)
            if population > num_family * 2:
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
            cur_cycles = []
            cur_num_clans = []
            cur_num_structures = []
            for society in societies:
                res = cluster([family.trait for family in society.families], [family.preference for family in society.families])
                cur_cycles.append(res[0])
                cur_num_clans.append(res[2])
                cur_num_structures.append(res[3])
            cur_cycle = sum(cur_cycles) / len(cur_cycles)
            cur_num_clan = sum(cur_num_clans) / len(cur_num_clans)
            cur_num_structure = sum(cur_num_structures) / len(cur_num_structures)
            cycle_ls.append(cur_cycle)
            num_clan_ls.append(cur_num_clan)
            num_structure_ls.append(cur_num_structure)
            # print(cur_cycle, cur_num_clan, cur_num_structure)

        num += 1
    if len(societies) == 0:
        structures = []

    if num == iter:
        if l == 0:
            k = 0
            for society in societies[:3]:
                flag = 0
                x, y = [family.trait for family in society.families], [family.preference for family in society.families]
                cycle, clusters, num_clans, num_structures = cluster(x, y)
                data = np.array([[family.trait, family.preference] for family in society.families])

                fig, ax = plt.subplots()

                for i in range(len(clusters)):
                    try:
                        ax.scatter(data[:, 0][clusters[i]], data[:, 1][clusters[i]], s=100-20*i, color=current_palette[i + 1])
                    except:
                        pass
                # axL.scatter(data[:,0], data[:,2], s=80)
                ax.set_xlabel(r"$t$",fontsize=24)
                ax.set_ylabel(r"$p$",fontsize=24)
                ax.tick_params(labelsize = 16)
                fig.tight_layout()
                fig.savefig(f"figs/structure_{cycle}_{path}_{k}.pdf", bbox_inches='tight')
                plt.close('all')

                # my_ls=[]
                # for i in range(1000):
                #     my_ls.extend([[i,society.df.iat[0,i][j],society.df.iat[1,i][j]] for j in range(len(society.df.iat[0,i]))])
                # df_res=pd.DataFrame(my_ls,columns=["time","t","p"])
                # fig, ax = plt.subplots()
                # ax.scatter(df_res["time"],df_res["t"], s=0.2, color= "blue")
                # ax.scatter(df_res["time"],df_res["p"], s=0.2, color= "red")
                # ax.set_xlabel("generation",fontsize=24)
                # ax.set_ylabel(r"$t, p$",fontsize=24)
                # ax.tick_params(labelsize = 16)
                # fig.tight_layout()
                # fig.savefig(f"figs/temporal_{cycle}_{path}_{k}.pdf", bbox_inches='tight')
                # plt.close('all')
                k += 1

    return cycle_ls, num_clan_ls, num_structure_ls


def run_simulation():
    df = pd.DataFrame(index=list(range(iter // 10)))
    df2 = pd.DataFrame(index=list(range(iter // 10)))
    df3 = pd.DataFrame(index=list(range(iter // 10)))
    k = 0
    if not os.path.exists(f"res/res_{path}.csv"):
        for l in range(100):
            try:
                res = main(l)
                if len(res) == 0:
                    continue
                else:
                    df[k] = res[0]
                    df2[k] = res[1]
                    df3[k] = res[2]
                    k += 1
            except:
                pass
        df.to_csv(f"res/res_{path}.csv")
        df2.to_csv(f"res/res2_{path}.csv")
        df3.to_csv(f"res/res3_{path}.csv")

#settings
num_family = 50
num_society = 30
mutation = 0.1
mutation = 0.03
r = 0.03
r = 50
iter = 1000
l = 0

mutation = 0.03
num_family = 100
for num_society in [[1, 2, 100],[3, 50], [5, 10, 30]][int(sys.argv[1]) % 3]:
    for r in [[0.01, 0.02], [0.03, 0.05], [0.1, 0.2], [0.3, 0.5], [1.0, 2.0], [3.0, 5.0], [10, 20], [30, 50], [100, 200]][(int(sys.argv[1]) // 3)]:
        path = f"{num_society}societies_{num_family}families_r{round(r * 1000)}pm_mutation{round(mutation * 1000)}pm"
        run_simulation()

