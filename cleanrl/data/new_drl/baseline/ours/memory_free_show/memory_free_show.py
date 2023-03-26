# 该文件绘制base290输入流下 四个模型的 dis precent 收敛情况 与两个baseline的横线对比图，四个模型分别对应四个子图，每张子图上一条曲线，两条横线，四张子图组成一个大图
# 并放置一个总的legend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv

def tensorboard_smoothing(values, smooth = 0.92):
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

base = ['210','290']

figure, ax = plt.subplots(2,1)

font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12,
            }
font2 = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 13,
            }

for i in range(2):
    labels = ax[i].get_xticklabels() + ax[i].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

for idx, m in enumerate(base):

    sns.set(style="whitegrid")
    mark = [50 * i + 10 for i in range(0, 11)]

    # 因为只有一个base的数据，所以长度设置为一个csv文件的数据长度：416个点
    total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(520)], 'x':[0 for _ in range(520)], 'color':[0 for _ in range(520)]})
    total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(520)], 'x':[0 for _ in range(520)], 'color':[0 for _ in range(520)]})

    total_df_origin_data_cnt = 0
    total_df_smoothed_data_cnt = 0


    path = "./base{}/".format(base[idx])
    files = os.listdir(path)
    print(path, files)

    for csv in files:
        if csv[-4:] != ".csv":
            continue
        df = pd.read_csv("./base{}/{}".format(base[idx], csv))
        smoothed = tensorboard_smoothing(df['Value'])
        df['smoothed'] = pd.Series(smoothed)
        df['x'] = df.index

        for i in range(520):
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed'] * 1024
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = "DRL"
            total_df_smoothed_data_cnt += 1

        for i in range(520):
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value'] * 1024
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = "DRL"
            total_df_origin_data_cnt += 1

    
    ax[idx].plot([i for i in range(len(list(total_df_origin_data['data'])))], list(total_df_origin_data['data']), color="#4C72B0", alpha=0.2)
    ax[idx].plot([i for i in range(len(list(total_df_smoothed_data['data'])))], list(total_df_smoothed_data['data']), color="#4C72B0", marker='o', 
             markersize=5, markeredgewidth=1, markevery=mark, alpha=1, label="Avg effective memory released per second")
    
    
    ax[idx].set_xticks([0, 104, 208, 312, 416, 520], ['0', '200k','400k','600k', '800k', '1000K'])
    ax[idx].grid(linestyle='-.')
    ax[idx].tick_params(labelsize=13)
    loc = 1
    if idx == 0:
        loc = 4
    ax[idx].legend(prop=font1, loc=loc, markerscale=1,)

    ax[idx].set_title(base[idx], fontdict=font2)
    ax[idx].set_xlabel('Step', fontdict=font1)
    ax[idx].set_ylabel('Memory Free(MB)', fontdict=font1)
    # plt.tight_layout()









plt.savefig("./memory_free.pdf")
plt.savefig("./memory_free.png")
plt.show()
print("-------------------------------")
