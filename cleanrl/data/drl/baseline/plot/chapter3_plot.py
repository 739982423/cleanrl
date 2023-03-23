# 该文件绘制第三章结论部分的图像
# 包括：在不同基准的输入流下DRL模型的收敛曲线(reward)，以及对应baseline的横线；不同基准输入流下的请求丢弃曲线，以及对应的baseline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

ascend_list = [i * 10 for i in range(1)]
drl_seeds = ['201', '202']
for ascend in ascend_list:
    df_summary = pd.DataFrame({'data':[0 for _ in range(1000 * len(drl_seeds))],
                               'smoothed':[0 for _ in range(1000 * len(drl_seeds))], 
                               'x':[0 for _ in range(1000 * len(drl_seeds))],
                               'gamma': [0 for _ in range(1000 * len(drl_seeds))]})
    df_summary['data'].astype('float')
    df_summary['x'].astype('int')
    cnt = 0
    for idx, drl_seed in enumerate(drl_seeds):
        df_drl_line1 = pd.read_csv("../ours/GPUcluster-1a_p_4m__ppo1a_4m__{}__r{}_v{}_d{}_m{}_r10___gamma0.99.csv".format(drl_seed,100+ascend,90+ascend,80+ascend,180+ascend))
        df_drl_line2 = pd.read_csv("../ours/GPUcluster-1a_p_4m__ppo1a_4m__{}__r{}_v{}_d{}_m{}_r10___gamma0.9.csv".format(drl_seed,100+ascend,90+ascend,80+ascend,180+ascend))
       
        df_drl_line1['idx'] = df_drl_line1.index
        df_drl_line2['idx'] = df_drl_line2.index

        smoothed1 = tensorboard_smoothing(df_drl_line1['Value'])
        smoothed2 = tensorboard_smoothing(df_drl_line2['Value'])

        df_drl_line1['smoothed'] = pd.Series(smoothed1)
        df_drl_line2['smoothed'] = pd.Series(smoothed2)

        for j in range(1000):
            # print(df_drl_line.loc[j]['smoothed'])
            value = df_drl_line1.loc[j]['Value']
            value_smoothed = df_drl_line1.loc[j]['smoothed']
            df_summary.loc[cnt, 'data'] = float(value)
            df_summary.loc[cnt, 'smoothed'] = float(value_smoothed)
            df_summary.loc[cnt, 'x'] = j
            df_summary.loc[cnt, 'gamma'] = 0.99
            cnt += 1

        for j in range(1000):
            value = df_drl_line2.loc[j]['Value']
            value_smoothed = df_drl_line2.loc[j]['smoothed']
            df_summary.loc[cnt, 'data'] = float(value)
            df_summary.loc[cnt, 'smoothed'] = float(value_smoothed)
            df_summary.loc[cnt, 'x'] = j
            df_summary.loc[cnt, 'gamma'] = 0.9
            cnt += 1

    df_baseline_1 = pd.read_csv("./res_ascent{}.csv".format(ascend))

    mark1 = [90 * i for i in range(0, 11)]
    mark2 = [[90 * i for i in range(0, 11)] for _ in range(2)]
    figure, ax = plt.subplots()
    sns.set(style="whitegrid")
    sns.set_palette('husl')

    sns.lineplot(x="x", y="smoothed", data=df_summary, style='gamma', dashes=False, hue='gamma', palette=sns.set_palette('husl'))
    sns.lineplot(x="x", y="data", data = df_baseline_1, marker='v', dashes=False, markersize=6, markeredgewidth=1.5, markevery=mark1, 
                alpha=1, markeredgecolor='none', label="MinWorkload", palette=sns.set_palette('husl'))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14,
             }
    plt.grid(axis='y')
    plt.tick_params(labelsize=13)
    plt.legend(prop=font1, loc=4, markerscale=1,)
    plt.savefig('./figure/base{}_seed_{}.pdf'.format(str(100+ascend), ''.join(drl_seeds)))
    plt.savefig('./figure/base{}_seed_{}.png'.format(str(100+ascend), ''.join(drl_seeds)))
    plt.show()
    break
