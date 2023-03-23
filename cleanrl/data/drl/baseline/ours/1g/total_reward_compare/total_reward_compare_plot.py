# 该文件绘制基准输入流大小从100-170，8种情况下的reward收敛对比图
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv

def tensorboard_smoothing(values, smooth = 0.88):
    """不需要传入step"""
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = smooth + 1
    x = values[0]
    res = [x]
    for i in range(1, len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


files = os.listdir()
figure, ax = plt.subplots()
sns.set(style="whitegrid")
sns.set_palette(sns.color_palette("RdYlBu", 8))
# sns.palplot(sns.color_palette())

total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(8000)], 'x':[0 for _ in range(8000)], 'color':[0 for _ in range(8000)]})
total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(8000)], 'x':[0 for _ in range(8000)], 'color':[0 for _ in range(8000)]})

total_df_origin_data_cnt = 0
total_df_smoothed_data_cnt = 0

for csv in files:
    if csv[-4:] != ".csv":
        continue
    df = pd.read_csv(csv)
    smoothed = tensorboard_smoothing(df['Value'])
    df['smoothed'] = pd.Series(smoothed)
    df['x'] = df.index

    for i in range(1000):
        total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed']
        total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
        total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = int(csv[36:39])
        total_df_smoothed_data_cnt += 1
    for i in range(1000):
        total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value']
        total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
        total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = int(csv[36:39])
        total_df_origin_data_cnt += 1

mark = [90 * i for i in range(0, 11)]

# palette=sns.cubehelix_palette(8)
# palette=sns.color_palette("RdYlBu", 8)
# palette=sns.hls_palette(8 , l = .5, s = .7)

sns.lineplot(x="x", y="data", hue='color', data=total_df_origin_data, palette=sns.color_palette("RdYlBu", 8), legend=False, alpha=0.3)
s = sns.lineplot(x="x", y="data", data=total_df_smoothed_data, style='color', hue='color', dashes=False, linestyle="-", markers=['o' for _ in range(8)], 
                 **dict(linewidth=2, ), markersize=6, markeredgewidth=1, markevery=mark, 
                 palette=sns.color_palette("RdYlBu", 8),  alpha=0.9)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }
plt.grid()
plt.tick_params(labelsize=13)
# ax.legend(loc='lower right')
plt.legend(prop=font1, loc=4, markerscale=1,)
plt.xlabel('Step', fontdict=font1)
plt.ylabel('Reward', fontdict=font1)
plt.tight_layout()
plt.show()
