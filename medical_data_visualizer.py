import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight']/((df['height']/100)**2) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

#print(df.iloc[40:50 ,7:9])
# print(df.head())

# 4
def draw_cat_plot():
    # print("in plot 1")
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # df_cat.info()

#     # 6
    # mask = df_cat['cardio'] == 0
    # df0, df1 = df_cat[mask], df_cat[~mask]
    # # df0, df1 = df_cat.loc[mask].drop('cardio', axis=1).groupby(by=["variable"]).sum().reset_index(), df_cat.loc[~mask].drop('cardio', axis=1).groupby(by=["variable"]).sum().reset_index().rename(columns={"value": "value1"})
    # # df_cat = pd.merge(df0, df1, how = 'left')
    
    # df0.info()
    # # print(df0)
    # df1.info()
    

#     # 7

#     # 8
    fig = sns.catplot(data=df_cat, x="variable", col="cardio", kind="count", hue = "value", errorbar = None)#, height=4, aspect=.6,)
    fig.set_ylabels("total")

#     # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    print("in plot 2")
#     # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

#     # 12
    corr = df_heat.corr()

    print(corr)
#     # 13
    mask = np.triu(np.ones_like(corr))

    # print(mask.shape)
    # print(corr.shape)

#     # 14
    fig, ax = plt.subplots()
      # 15
    sns.set(font_scale=0.5)
     
    plot = sns.heatmap(corr, mask = mask, cbar = True, annot = True, annot_kws={"size": 5}, linewidths=0.05, linecolor='white', center = 0, fmt = '.1f', cbar_kws = {'shrink': 0.5, 'aspect': 20,'ticks': [-0.08, -0, 0.08, 0.16, 0.24]}, vmin=-0.14, vmax=0.3)
    plot.tick_params(axis='both', labelsize=5, length = 2, pad = 1, width = 0.5)

    # plot.set_ylabel("Tip ($)", fontsize=14)
    fig = plot.figure

    # fig
#     # 16
    fig.savefig('heatmap.png')
    return fig
