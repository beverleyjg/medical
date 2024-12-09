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

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6
    # 7
    # 8
    f = sns.catplot(data=df_cat, x="variable", col="cardio", kind="count", hue = "value", errorbar = None)
    f.set_axis_labels("variable","total")
    fig = f.figure
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    print("in plot 2")
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    print(corr)
    # 13
    mask = np.triu(np.ones_like(corr))

    # 14
    fig, ax = plt.subplots()
    # 15
    sns.set(font_scale=0.5)
     
    plot = sns.heatmap(corr, mask = mask, cbar = True, annot = True, annot_kws={"size": 5}, linewidths=0.05, linecolor='white', center = 0, fmt = '.1f', cbar_kws = {'shrink': 0.5, 'aspect': 20,'ticks': [-0.08, -0, 0.08, 0.16, 0.24]}, vmin=-0.14, vmax=0.3)
    plot.tick_params(axis='both', labelsize=5, length = 2, pad = 1, width = 0.5)
    fig = plot.figure

    # 16
    fig.savefig('heatmap.png')
    return fig
