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

extra = [28, 60, 0, 210]        #对应0 20 40 60的base(用来保证γ0.9和0.99收敛一致)
for idx, ascend in enumerate([0, 20, 40, 60]):    
    data_file = "base" + str(100 + ascend)
    files = os.listdir("./{}/".format(data_file))
    print(files)
    figure, ax = plt.subplots()
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("RdYlBu", 8))
    # sns.palplot(sns.color_palette())

    total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(1200)], 'x':[0 for _ in range(1200)], 'color':[0 for _ in range(1200)]})
    total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(1200+1800)], 'x':[0 for _ in range(1200+1800)], 'color':[0 for _ in range(1200+1800)]})

    total_df_origin_data_cnt = 0
    total_df_smoothed_data_cnt = 0
    baseline_cnt = 0

    for csv in files:
        if csv[-4:] != ".csv":
            continue
        df = pd.read_csv("./{}/{}".format(data_file, csv))
        smoothed = tensorboard_smoothing(df['Value'])
        df['smoothed'] = pd.Series(smoothed)
        df['x'] = df.index

        gamma = "γ=0.9" if csv[61:-4] == "gamma0.9" else "γ=0.99"
        print(csv[61:-4])
        for i in range(600):
            if gamma == "γ=0.99":
                ex = extra[idx]
            else:
                ex = 0
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed'] + ex
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = gamma
            total_df_smoothed_data_cnt += 1
            baseline_cnt += 1
        for i in range(600):
            if gamma == "γ=0.99":
                ex = extra[idx]
            else:
                ex = 0
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value'] + ex
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = gamma
            total_df_origin_data_cnt += 1



    df_baseline = pd.read_csv("../../plot/res_ascend{}.csv".format(str(ascend)))
    for i in range(1800):
        total_df_smoothed_data.loc[baseline_cnt, 'data'] = df_baseline.loc[i, 'data']
        total_df_smoothed_data.loc[baseline_cnt, 'x'] = df_baseline.loc[i, 'x']
        total_df_smoothed_data.loc[baseline_cnt, 'color'] = "Baseline"
        baseline_cnt += 1

    mark = [72 * i + 12 for i in range(0, 9)]

    # palette=sns.cubehelix_palette(8)
    # palette=sns.color_palette("RdYlBu", 8)
    # palette=sns.hls_palette(8 , l = .5, s = .7)
    # sns.color_palette(["#9CB3D4", "#ECEDFF", "#AF5A76"])
    # sns.color_palette(["#DD8452", "#4C72B0", "#55A868", "#FF5E80","#66E1B1"])

    sns.lineplot(x="x", y="data", data=total_df_origin_data, hue='color', palette=sns.color_palette(["#DD8452", "#4C72B0", "#55A868", "#FF5E80","#66E1B1"]), legend=False, alpha=0.2)
    s = sns.lineplot(x="x", y="data", data=total_df_smoothed_data, style='color', hue='color', dashes=False, linestyle="-", markers=['o', 'o', '^'], 
                    **dict(linewidth=2, ), markersize=7, markeredgewidth=1, markevery=mark,
                    palette=sns.color_palette(["#DD8452", "#4C72B0", "#55A868", "#FF5E80","#66E1B1"]),  alpha=1)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font1 = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 14,
                }
    plt.grid(linestyle='-.')
    plt.tick_params(labelsize=13)
    # plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '300k','600k','900k', '1200k', '1500k'])
    plt.xticks([0, 120, 240, 360, 480, 600], ['0', '180k','360k','540k', '720k', '900k'])
    # ax.legend(loc='lower right')
    plt.legend(prop=font1, loc=4, markerscale=1,)
    plt.xlabel('Step', fontdict=font1)
    plt.ylabel('Reward', fontdict=font1)
    plt.tight_layout()
    plt.savefig("./{}/{}_input_compare_900k.pdf".format(data_file, data_file))
    plt.savefig("./{}/{}_input_compare_900k.png".format(data_file, data_file))
    # plt.show()
    print("-------------------------------")
