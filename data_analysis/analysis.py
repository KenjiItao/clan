import pandas as pd
import scipy.stats
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
import japanize_matplotlib

sns.set(style='whitegrid')
current_palette = sns.color_palette("colorblind", 5)
if True:
    current_palette[0] = (0 / 255, 114 / 255, 178 / 255)
    current_palette[1] = (240 / 255, 228 / 255, 66 / 255)
    current_palette[2] = (0 / 255, 158 / 255, 115 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)

def rand_jitter(arr):
    stdev = .03 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def structure(df):
    res = pd.Series()
    for id in df.index:
        cur = df.loc[id]
        if cur["SCCS222"] == 1 and cur["SCCS224"] == 1 and cur["SCCS219"]< 3:
            structure = 1
        elif cur["SCCS222"] > 1 and cur["SCCS224"] > 1:
            structure = 4
        elif cur["SCCS222"] == 1 and cur["SCCS224"] == 6:
            structure = 2
        elif cur["SCCS222"] == 6 and cur["SCCS224"] == 1:
            structure = 2
        elif cur["SCCS70"] == 3 and cur["SCCS224"] == 1 and cur["SCCS230"] == 1:
            structure = 2
        elif cur["SCCS70"] == 1 and cur["SCCS222"]== 1 and cur["SCCS230"] == 1:
            structure = 2
        elif cur["SCCS230"] == 2 and cur["SCCS222"] > 1:
            structure = 3
        elif cur["SCCS230"] == 3 and cur["SCCS224"] > 1:
            structure = 3
        elif cur["SCCS70"] == 3 and cur["SCCS230"] == 2:
            structure = 3
        elif cur["SCCS70"] == 1 and cur["SCCS230"] == 3:
            structure = 3
        else:
            structure = 0
        res[id] = structure
    return res

def correlation_analysis(data_pivot):
    var_sample.index = var_sample["id"]
    df_structure = data_pivot[data_pivot["structure2"] > 1]
    id_ls = var_sample["id"].tolist()
    id_ls.append("structure2")
    df_structure = df_structure[df_structure.columns & id_ls]

    df_structure.replace(88, np.nan, inplace = True)
    df_structure.replace(99, np.nan, inplace = True)

    for structure_ in [2, 3, 4]:
        df_structure[structure_] = 1 - 1 * (df_structure["structure2"] == structure_)

    df_res0 = pd.DataFrame(0.0, index = df_structure.columns[:-4], columns = [2,3, 4])

    for structure_ in [2, 3, 4]:
        res = pd.DataFrame(index = ["corr.", "p"])
        for col in df_structure.columns:
            df2 = df_structure[df_structure[structure_] == 1][["structure2", col]].dropna()
            x = df2["structure2"].values
            y = df2[col].values
            a, b = spearmanr(np.ravel(x), np.ravel(y))
            if b > 0:
                res[col] = [a, b]

        df_res0[structure_] += res.T["corr."]


    df_res0["sum"] = abs(df_res0.fillna(0)).sum(axis = 1) / 3
    df_res0["title"] = var_sample.loc[df_res0.index].title
    df_res0 = df_res0.sort_values("sum", ascending =  False)
    df_res0["null"] = df_structure.isnull().sum()
    df_res0["null"] = round((len(df_structure.index) - df_res0["null"]) / len(df_structure.index), 2)
    df_res = df_res0[df_res0["null"] > 0.1]
    id_ls = df_res.index.tolist()
    df_agg = np.round(df_structure.groupby("structure2").mean()[id_ls].T, 2)
    df_agg[["corr.", "title", "ratio"]] = df_res.loc[df_agg.index][["sum", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 2,  3, 4, "corr.", "ratio"])
    df_agg.columns = ["title", "dual",  "generalized", "restricted", "corr.", "ratio"]
    df_agg.to_csv("variables/variables_high_corr2.csv")

def normalization(data_pivot):
    df = data_pivot.copy()
    df.replace(88, np.nan, inplace = True)
    df.replace(99, np.nan, inplace = True)
    df = (df - df.mean()) / df.std(ddof=0)
    df["structure2"] = data_pivot["structure2"]
    df["descent"] = data_pivot["SCCS70"]
    data_structure = df[df["structure2"] > 0]
    data_structure["structure"] = data_structure["structure2"]
    data_structure["structure"].replace([1,2,3,4], ["incest", "dual", "generalized", "restricted"], inplace = True)
    data_structure.sort_values("structure2", inplace = True)
    return data_structure

def calc_parameters(data_structure):
    data_structure["d_c1"] =  data_structure["SCCS1772"]
    data_structure["d_c2"] =  data_structure["SCCS1737"]
    data_structure["d_c3"] =  data_structure["SCCS1770"]
    data_structure["d_c4"] =  data_structure["SCCS791"]
    data_structure["d_c5"] =  - data_structure["SCCS905"]
    data_structure["d_c_count"] = data_structure[["d_c1", "d_c2", "d_c3","d_c4", "d_c5"]].isnull().sum(axis = 1)
    data_structure[r"$d_c$"] = (data_structure["d_c1"].fillna(0) + data_structure["d_c2"].fillna(0) + data_structure["d_c3"].fillna(0)+ data_structure["d_c4"].fillna(0) + data_structure["d_c5"].fillna(0)) / (5 - data_structure["d_c_count"])

    data_structure["d_m1"] =  data_structure["SCCS173"]
    data_structure["d_m2"] =  - data_structure["SCCS782"]
    data_structure["d_m3"] =   - data_structure["SCCS768"]
    data_structure["d_m4"] =  data_structure["SCCS960"]
    data_structure["d_m5"] =  data_structure["SCCS962"]
    data_structure["d_m_count"] = data_structure[["d_m1", "d_m2", "d_m3", "d_m4", "d_m5"]].isnull().sum(axis = 1)
    data_structure[r"$d_m$"] = (data_structure["d_m1"].fillna(0) + data_structure["d_m2"].fillna(0) + data_structure["d_m3"].fillna(0) + data_structure["d_m4"].fillna(0) + data_structure["d_m5"].fillna(0) ) / (5 - data_structure["d_m_count"])

    return data_structure

def plot(data_structure):
    data_structure[r"$d_c$2"] = rand_jitter(data_structure[(data_structure[r"$d_c$"] > -100) & (data_structure[r"$d_m$"] > -100)][r"$d_c$"])
    data_structure[r"$d_m$2"] = rand_jitter(data_structure[(data_structure[r"$d_c$"] > -100) & (data_structure[r"$d_m$"] > -100)][r"$d_m$"])
    data_structure[r"$d_c$2"] = data_structure[r"$d_c$2"] - data_structure[r"$d_c$2"].min()
    data_structure[r"$d_m$2"] = data_structure[r"$d_m$2"] - data_structure[r"$d_m$2"].min()

    plt.figure()
    ax = sns.scatterplot(data = data_structure[data_structure["structure2"] > 0], x = r"$d_c$2", y = r"$d_m$2", hue = "structure2", s = 100, palette =  current_palette[1:])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_kinship.pdf", bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.scatterplot(data = data_structure[data_structure["structure2"] > 1], x = r"$d_c$2", y = r"$d_m$2", hue = "structure2",s = 100, palette =  current_palette[2:])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_kinship_wo_incest.pdf", bbox_inches='tight')
    plt.close('all')

    df = data_structure[data_structure["descent"] < 4]

    df["descent"].replace([1,2,3], ["matrilineal", "bilateral", "patrilineal"], inplace = True)
    plt.figure()
    ax = sns.histplot(data = df, x = r"$d_m$2", hue = "descent", multiple="stack", alpha = 1, hue_order=["bilateral", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    ax.set_xlabel(r"$\widetilde{d_m}$", fontsize = 24)
    plt.xticks(rotation=0)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_descent_bar.pdf", bbox_inches='tight')
    plt.close('all')

    df = data_structure[(data_structure["descent"] < 4) & (data_structure["structure2"] > 1)]
    df["descent"].replace([1,2,3], ["matrilineal", "double", "patrilineal"], inplace = True)
    df.sort_values("structure2", inplace = True)
    df["structure"].replace([2,3,4], ["dual", "generalized", "restricted"], inplace = True)
    plt.figure()
    ax = sns.histplot(data = df, x = "structure", hue = "descent", multiple="stack", alpha = 1, hue_order=["double", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    ax.get_legend().remove()
    ax.tick_params(labelsize=16)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"structure_descent2.pdf", bbox_inches='tight')
    plt.close('all')


data_whole = pd.read_csv("data/data.csv")
var_whole = pd.read_csv("data/variables.csv")
var_sample = pd.read_csv("data/variables_sample.csv")

data_pivot = data_whole.pivot_table(index = "soc_id", columns = "var_id", values="code")
data_pivot["structure2"] = structure(data_pivot)
correlation_analysis(data_pivot)
data_structure = normalization(data_pivot)
data_structure = calc_parameters(data_structure)
plot(data_structure)
