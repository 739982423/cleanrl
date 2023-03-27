# 该文件绘制基准输入流大小从210 230 250 270 290 390，6种情况下总reward对比
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv

def tensorboard_smoothing(values, smooth = 0.9):
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

model_name = "Reward"

total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(520)], 'x':[0 for _ in range(520)], 'Gamma':[0 for _ in range(520)]})
total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(520)], 'x':[0 for _ in range(520)], 'Gamma':[0 for _ in range(520)]})

figure, ax = plt.subplots()
sns.set(style="whitegrid")




total_df_origin_data_cnt = 0
total_df_smoothed_data_cnt = 0

for gamma in ['0.9', '0.95']:
    files = os.listdir("./gamma{}/".format(gamma))
    print(files)
    for csv in files:
        if csv[-4:] != ".csv":
            continue

        df = pd.read_csv("./gamma{}/{}".format(gamma, csv))
        smoothed = tensorboard_smoothing(df['Value'])
        df['smoothed'] = pd.Series(smoothed)
        df['x'] = df.index
        df['Gamma'] = gamma


        for i in range(len(df)):
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'Gamma'] = "Gamma=" + gamma
            total_df_smoothed_data_cnt += 1

        for i in range(len(df)):
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'Gamma'] = "Gamma=" + gamma
            total_df_origin_data_cnt += 1

mark = [50 * i + 10 for i in range(0, 11)]


s = sns.lineplot(x="x", y="data", data=total_df_origin_data, style='Gamma', hue='Gamma', dashes=False, linestyle="-", alpha=0.3, legend=False)
s = sns.lineplot(x="x", y="data", data=total_df_smoothed_data, style='Gamma', hue='Gamma', dashes=False, linestyle="-", 
                markers=['o'], **dict(linewidth=2, ), markersize=7, markeredgewidth=1, markevery=mark, alpha=1)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }

plt.grid(linestyle='-.')
plt.tick_params(labelsize=13)

plt.xticks([0, 104, 208, 312, 416, 520], ['0', '200k','400k','600k', '800k', '1000K'])
plt.yticks([0, -2000, -4000, -6000, -8000], ['0', '-2k','-4k','-6k', '-8k'])
# # ax.legend(loc='lower right')
plt.legend(prop=font1, loc=4, markerscale=1,)
plt.xlabel('Step', fontdict=font1)
plt.ylabel('Reward', fontdict=font1)
plt.tight_layout()
plt.savefig("./{}_compare_1M.pdf".format(model_name))
plt.savefig("./{}_compare_1M.png".format(model_name))
plt.show()
# print("-------------------------------")
