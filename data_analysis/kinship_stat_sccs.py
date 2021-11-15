import pandas as pd
import scipy.stats
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
import japanize_matplotlib
import geopandas as gpd
import json
from shapely.geometry import Point

def structure(df):
    res = pd.Series()
    for id in df.index:
        cur = df.loc[id]
        if cur["SCCS222"] > 1 and cur["SCCS224"] > 1:
            structure = 4
        elif cur["SCCS70"] == 2 and cur["SCCS230"] == 1:
            structure = 4
        elif cur["SCCS70"] == 2 and cur["SCCS231"] == 5:
            structure = 4
        elif cur["SCCS222"] == 1 and cur["SCCS224"] == 6:
            structure = 2
        elif cur["SCCS222"] == 6 and cur["SCCS224"] == 1:
            structure = 2
        elif cur["SCCS70"] == 3 and cur["SCCS222"] > 1 and cur["SCCS230"] == 1:
            structure = 2
        elif cur["SCCS70"] == 1 and cur["SCCS224"] > 1 and cur["SCCS230"] == 1:
            structure = 2
        elif cur["SCCS230"] == 2 and cur["SCCS222"] > 1:
            structure = 3
        elif cur["SCCS230"] == 3 and cur["SCCS224"] > 1:
            structure = 3
        elif cur["SCCS70"] == 3 and cur["SCCS230"] == 2:
            structure = 3
        elif cur["SCCS70"] == 1 and cur["SCCS230"] == 3:
            structure = 3
        elif cur["SCCS231"] == 6 and cur["SCCS222"] > 1:
            structure = 3
        elif cur["SCCS231"] == 1 and cur["SCCS224"] > 1:
            structure = 3
        elif cur["SCCS222"] == 1 and cur["SCCS224"] == 1:
            structure = 1
        else:
            structure = 0
        res[id] = structure
    return res

def correlation_analysis(data_pivot):
    var_sample.index = var_sample["id"]
    df_structure = data_pivot[data_pivot["structure2"] > 0]
    id_ls = var_sample["id"].tolist()
    id_ls.append("structure2")
    df_structure = df_structure[df_structure.columns & id_ls]

    df_structure.replace(88, np.nan, inplace = True)
    df_structure.replace(99, np.nan, inplace = True)

    df_structure[["12", "13", "14", "23", "24", "34"]] = 0
    for ind in df_structure.index:
        if df_structure.loc[ind, "structure2"] == 1:
            df_structure.loc[ind, ["12", "13", "14", "23", "24", "34"]] = [1, 1, 1, 0, 0, 0]
        if df_structure.loc[ind, "structure2"] == 2:
            df_structure.loc[ind, ["12", "13", "14", "23", "24", "34"]] = [1, 0, 0, 1, 1, 0]
        if df_structure.loc[ind, "structure2"] == 3:
            df_structure.loc[ind, ["12", "13", "14", "23", "24", "34"]] = [0, 1, 0, 1, 0, 1]
        if df_structure.loc[ind, "structure2"] == 4:
            df_structure.loc[ind, ["12", "13", "14", "23", "24", "34"]] = [0, 0, 1, 0, 1, 1]

    df_res0 = pd.DataFrame(0.0, index = df_structure.columns[:-7], columns = ["12", "13", "14", "23", "24", "34"])

    for structure_ in ["23", "24", "34"]:
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
    df_res = df_res0[df_res0["null"] >= 0.1]
    id_ls = df_res.index.tolist()
    df_agg = np.round(df_structure.groupby("structure2").mean()[id_ls].T, 2)
    df_agg[["corr.", "title", "ratio"]] = df_res.loc[df_agg.index][["sum", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 1, 2,  3, 4, "corr.", "ratio"])
    df_agg.columns = ["title", "incest","dual",  "generalized", "restricted", "corr.", "ratio"]
    df_agg.to_csv("variables/variables_high_corr_wo_incest.csv")


    df_res0 = pd.DataFrame(0.0, index = df_structure.columns[:-7], columns = ["12", "13", "14", "23", "24", "34"])

    for structure_ in ["12", "13", "14", "23", "24", "34"]:
    # for structure_ in ["23", "24", "34"]:
        res = pd.DataFrame(index = ["corr.", "p"])
        for col in df_structure.columns:
            df2 = df_structure[df_structure[structure_] == 1][["structure2", col]].dropna()
            x = df2["structure2"].values
            y = df2[col].values
            a, b = spearmanr(np.ravel(x), np.ravel(y))
            if b > 0:
                res[col] = [a, b]

        df_res0[structure_] += res.T["corr."]

    df_res0["sum"] = abs(df_res0.fillna(0)).sum(axis = 1) / 6
    df_res0["title"] = var_sample.loc[df_res0.index].title
    df_res0 = df_res0.sort_values("sum", ascending =  False)
    df_res0["null"] = df_structure.isnull().sum()
    df_res0["null"] = round((len(df_structure.index) - df_res0["null"]) / len(df_structure.index), 2)
    df_res = df_res0[df_res0["null"] >= 0.1]
    id_ls = df_res.index.tolist()
    df_agg = np.round(df_structure.groupby("structure2").mean()[id_ls].T, 2)
    df_agg[["corr.", "title", "ratio"]] = df_res.loc[df_agg.index][["sum", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 1, 2,  3, 4, "corr.", "ratio"])
    df_agg.columns = ["title", "incest","dual",  "generalized", "restricted", "corr.", "ratio"]
    df_agg.to_csv("variables/variables_high_corr_w_incest.csv")

def correlation_analysis_descent(data_pivot):
    var_sample.index = var_sample["id"]
    df_structure = data_pivot[data_pivot["SCCS70"] < 4]
    id_ls = var_sample["id"].tolist()

    df_structure = df_structure[df_structure.columns & id_ls]

    df_structure.replace(88, np.nan, inplace = True)
    df_structure.replace(99, np.nan, inplace = True)


    for descent in [1, 2, 3]:
        df_structure[descent] = 1 - 1 * (df_structure["SCCS70"] == descent)

    df_res0 = pd.DataFrame(0.0, index = df_structure.columns[:-4], columns = [1, 2, 3])

    for descent in [1, 2, 3]:
        res = pd.DataFrame(index = ["corr.", "p"])
        for col in df_structure.columns:
            df2 = df_structure[df_structure[descent] == 1][["SCCS70", col]].dropna()
            x = df2["SCCS70"].values
            y = df2[col].values
            a, b = spearmanr(np.ravel(x), np.ravel(y)) # リストを整形し相関係数:aとp値:bの計算
            if b > 0:
                res[col] = [a, b]

        df_res0[descent] += res.T["corr."]

    df_res0["sum"] = abs(df_res0[[1,2,3]].fillna(0)).sum(axis = 1) / 3
    df_res0["title"] = var_sample.loc[df_res0.index].title
    df_res0 = df_res0.sort_values("sum", ascending =  False)
    df_res0["null"] = df_structure.isnull().sum()
    df_res0["null"] = round((len(df_structure.index) - df_res0["null"]) / len(df_structure.index), 2)
    df_res = df_res0[df_res0["null"] > 0.1]
    id_ls = df_res.index.tolist()
    id_ls.remove("SCCS70")
    df_agg = np.round(df_structure.groupby("SCCS70").mean()[id_ls].T, 2)
    df_agg[["corr.", "title", "ratio"]] = df_res.loc[df_agg.index][["sum", "title", "null"]]
    df_agg = df_agg.reindex(columns = ["title", 1, 2,  3, "corr.", "ratio"])
    df_agg.columns = ["title", "materenal",  "bilateral", "paternal", "corr.", "ratio"]
    df_agg.to_csv("variables/descent_variables_high_corr.csv")

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
    data_structure["d_c1"] =  data_structure["SCCS1120"]
    # data_structure["d_c1"] =  data_structure["SCCS791"]
    data_structure["d_c2"] =  data_structure["SCCS1770"]
    # data_structure["d_c1"] =  data_structure["SCCS1120"]
    data_structure["d_c3"] =  data_structure["SCCS1772"]
    # data_structure["d_c4"] =  - data_structure["SCCS905"]
    data_structure["d_c4"] =  data_structure["SCCS1737"]
    data_structure["d_c5"] =  data_structure["SCCS788"]
    data_structure["d_c_count"] = data_structure[["d_c1", "d_c2", "d_c3","d_c4", "d_c5"]].isnull().sum(axis = 1)
    data_structure[r"$d_c$"] = (data_structure["d_c1"].fillna(0) + data_structure["d_c2"].fillna(0) + data_structure["d_c3"].fillna(0)+ data_structure["d_c4"].fillna(0) + data_structure["d_c5"].fillna(0)) / (5 - data_structure["d_c_count"])

    data_structure["d_m1"] =  data_structure["SCCS173"]
    data_structure["d_m2"] =  data_structure["SCCS960"]
    # data_structure["d_m2"] =  data_structure["SCCS962"]
    # data_structure["d_m2"] =  - data_structure["SCCS773"]
    data_structure["d_m3"] =  data_structure["SCCS961"]
    data_structure["d_m4"] =  - data_structure["SCCS782"]
    data_structure["d_m5"] =   - data_structure["SCCS768"]
    data_structure["d_m_count"] = data_structure[["d_m1", "d_m2", "d_m3", "d_m4", "d_m5"]].isnull().sum(axis = 1)
    data_structure[r"$d_m$"] = (data_structure["d_m1"].fillna(0) + data_structure["d_m2"].fillna(0) + data_structure["d_m3"].fillna(0) + data_structure["d_m4"].fillna(0) + data_structure["d_m5"].fillna(0) ) / (5 - data_structure["d_m_count"])

    return data_structure

def rand_jitter(arr):
    stdev = .03 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def structure_plot(data_structure):
    sns.set(style='whitegrid')
    data_structure[r"$d_c$2"] = rand_jitter(data_structure[(data_structure[r"$d_c$"] > -10) & (data_structure[r"$d_m$"] > -10)][r"$d_c$"])
    data_structure[r"$d_m$2"] = rand_jitter(data_structure[(data_structure[r"$d_c$"] > -10) & (data_structure[r"$d_m$"] > -10)][r"$d_m$"])
    data_structure[r"$d_c$2"] = data_structure[r"$d_c$2"] - data_structure[r"$d_c$2"].min()
    data_structure[r"$d_m$2"] = data_structure[r"$d_m$2"] - data_structure[r"$d_m$2"].min()

    plt.figure()
    # ax = fig.add_subplot(111, aspect=2)
    # ax = sns.scatterplot(data = data_structure[data_structure["structure2"] > 0], x = r"$d_c$2", y = r"$d_m$2", hue = "structure2", s = 100, palette =  current_palette[1:])
    ax = sns.scatterplot(data = data_structure[data_structure["structure2"] == 1], x = r"$d_c$2", y = r"$d_m$2", s = 100, c =  current_palette[1])
    ax = sns.scatterplot(data = data_structure[data_structure["structure2"] > 1], x = r"$d_c$2", y = r"$d_m$2", hue = "structure2", s = 100, palette =  current_palette[2:])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    # ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_kinship.pdf", bbox_inches='tight')
    plt.close('all')

    plt.figure()
    # ax = fig.add_subplot(111, aspect=2)
    ax = sns.scatterplot(data = data_structure[data_structure["structure2"] > 1], x = r"$d_c$2", y = r"$d_m$2", hue = "structure2",s = 100, palette =  current_palette[2:])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    # ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_kinship_wo_incest.pdf", bbox_inches='tight')
    plt.close('all')

    df = data_structure[data_structure["descent"] < 4]
    # df["descent"].replace([1,2,3], ["母系", "双系", "父系"], inplace = True)

    plt.figure()
    ax = sns.scatterplot(data = df, x = r"$d_c$2", y = r"$d_m$2", hue = "descent", s = 80, palette = [current_palette[0], current_palette[1], current_palette[3]])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_descent.pdf", bbox_inches='tight')
    plt.close('all')

    df["descent"].replace([1,2,3], ["matrilineal", "bilateral", "patrilineal"], inplace = True)
    plt.figure()
    ax = sns.histplot(data = df, x = r"$d_m$2", hue = "descent", multiple="stack", alpha = 1, hue_order=["bilateral", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    # ax=sns.heatmap(df_fig, vmin = -1, vmax = 2, cmap = sns.color_palette("rocket", as_cmap=True), cbar = True, square = True)
    # ax=sns.heatmap(df_fig,vmin=-0.1,vmax=n-0.9,cmap="Greys",square=True)
    ax.set_xlabel(r"$\widetilde{d_m}$", fontsize = 24)
    plt.xticks(rotation=0)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_descent_bar.pdf", bbox_inches='tight')
    plt.close('all')

    df = data_structure[(data_structure["descent"] < 4) & (data_structure["structure2"] > 0)]
    # df["descent"].replace([1,2,3], ["母系", "双系", "父系"], inplace = True)

    plt.figure()
    ax = sns.scatterplot(data = df, x = r"$d_c$2", y = r"$d_m$2", hue = "descent", s = 80, palette = [current_palette[0], current_palette[1], current_palette[3]])
    ax.set_xlabel(r"$\widetilde{d_c}$", fontsize=20)
    ax.set_ylabel(r"$\widetilde{d_m}$", fontsize=20)
    ax.set_xlim((-0.2, 4.0))
    ax.set_ylim((-0.2, 4.0))
    ax.tick_params(labelsize=12)
    ax.get_legend().remove()
    ax.set_aspect('equal', adjustable='box')
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"phase_descent_2.pdf", bbox_inches='tight')
    plt.close('all')

def kinship_plot(data_pivot):
    geo_df = gpd.GeoDataFrame(index = ["type", "name", "marker-color", "marker-size", "geometry"])
    df_structure = data_pivot[(data_pivot["structure2"] > 0)]

    for key in df_structure.index:
        geo_df[len(geo_df.columns)] = ["Feature", key, cur_pal[df_structure.at[key, "structure2"]], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]

    geo_df = geo_df.T
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    map_df.plot(ax = ax, color = "grey")
    # geo_df.plot(ax = ax, color = geo_df["marker-color"], markersize = 30, marker = "^")
    geo_df[geo_df["marker-color"] == cur_pal[1]].plot(ax = ax, color = geo_df[geo_df["marker-color"] == cur_pal[1]]["marker-color"], markersize = 30, marker = "^")
    geo_df[geo_df["marker-color"] != cur_pal[1]].plot(ax = ax, color = geo_df[geo_df["marker-color"] != cur_pal[1]]["marker-color"], markersize = 30, marker = "^")
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig("kinship_worldmap.pdf", bbox_inches='tight')

def descent_plot(data_pivot):
    geo_df = gpd.GeoDataFrame(index = ["type", "name", "marker-color", "marker-size", "geometry"])
    df_structure = data_pivot[(data_pivot["descent"] < 4)]

    for key in df_structure.index:
        if round(df_structure.at[key, "descent"]) == 1:
            geo_df[len(geo_df.columns)] = ["Feature", key, cur_pal[0], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
        elif round(df_structure.at[key, "descent"]) == 2:
            geo_df[len(geo_df.columns)] = ["Feature", key, cur_pal[1], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
        elif round(df_structure.at[key, "descent"]) == 3:
            geo_df[len(geo_df.columns)] = ["Feature", key, cur_pal[3], "small", Point([tdwg[key]["lon"], tdwg[key]["lat"]])]

    geo_df = geo_df.T

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    map_df.plot(ax = ax, color = "grey")
    geo_df.plot(ax = ax, color = geo_df["marker-color"], markersize = 30, marker = "^")
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig("descent_worldmap.pdf", bbox_inches='tight')

def structure_descent(data_structure):
    sns.set(style='whitegrid')
    df = data_structure[(data_structure["descent"] < 4) & (data_structure["structure2"] > 1)]
    # df["descent"].replace([1,2,3], ["母系", "双系", "父系"], inplace = True)
    df["descent2"] = df["descent"]
    df["descent"].replace([1,2,3], ["matrilineal", "double", "patrilineal"], inplace = True)
    df.sort_values("structure2", inplace = True)
    df["structure"].replace([2,3,4], ["dual", "generalized", "restricted"], inplace = True)
    plt.figure()
    ax = sns.histplot(data = df, x = "structure", hue = "descent", multiple="stack", alpha = 1, hue_order=["double", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"structure_descent.pdf", bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.histplot(data = df, x = "structure", hue = "descent", multiple="stack", alpha = 1, hue_order=["double", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    ax.get_legend().remove()
    # ax.get_xlabel().remove()
    ax.tick_params(labelsize=16)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"structure_descent2.pdf", bbox_inches='tight')
    plt.close('all')

    df_fig = pd.crosstab(df.structure2, df.descent2)
    df_fig.index = ["dual", "generalized", "restricted"]
    df_fig.columns = ["matrilineal", "double", "patrilineal"]
    df_fig = df_fig.T
    df_fig = df_fig / df_fig.sum(axis = 0)
    df_fig = df_fig.T
    df_fig = df_fig.reindex(columns = ["matrilineal", "patrilineal", "double"])

    plt.figure()
    # ax = sns.histplot(data = df, x = "structure", hue = "descent", stat = "density", common_bins = False, multiple="stack", alpha = 1, hue_order=["double", "patrilineal", "matrilineal"], palette = [current_palette[2], current_palette[3], current_palette[1]])
    ax = df_fig.plot.bar(stacked = True, color = ["#f0e442", "#d55e00","#009e73"])
    ax.get_legend().remove()
    plt.xticks(rotation=0)
    # ax.get_xlabel().remove()
    ax.tick_params(labelsize=16)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"structure_descent3.pdf", bbox_inches='tight')
    plt.close('all')

current_palette = sns.color_palette("colorblind", 5)
if True:
    current_palette[0] = (0 / 255, 114 / 255, 178 / 255)
    current_palette[1] = (240 / 255, 228 / 255, 66 / 255)
    current_palette[2] = (0 / 255, 158 / 255, 115 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)
    cur_pal = current_palette.as_hex()

data_whole = pd.read_csv("data/data.csv")
var_whole = pd.read_csv("data/variables.csv")
var_sample = pd.read_csv("data/variables_sample.csv")
geo_dir = "../../dplace-data-master/geo"
map_df = gpd.read_file(os.path.join(geo_dir,'level2.json'))

tdwg_open = open(os.path.join(geo_dir,'societies_tdwg.json'), 'r')
tdwg = json.load(tdwg_open)

data_pivot = data_whole.pivot_table(index = "soc_id", columns = "var_id", values="code")
data_pivot["structure2"] = structure(data_pivot)
correlation_analysis(data_pivot)
data_structure = normalization(data_pivot)
data_structure = calc_parameters(data_structure)
structure_plot(data_structure)
structure_descent(data_structure)
kinship_plot(data_pivot)
data_pivot["descent"] = data_pivot["SCCS70"]
descent_plot(data_pivot)

data_pivot["structure2"].value_counts()
data_pivot["descent"].value_counts()
data_structure[(data_structure[r"$d_c$"] > -100) & (data_structure[r"$d_m$"] > -100)]["structure2"].value_counts()
