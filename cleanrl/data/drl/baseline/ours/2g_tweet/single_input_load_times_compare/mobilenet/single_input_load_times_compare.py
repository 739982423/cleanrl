# 该文件绘制基准输入流大小从210 230 250 270 290 390，6种情况下的MobileNet_v2模型加载次数对比图
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

model_name = "MobileNet_v2"

total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(2496)], 'x':[0 for _ in range(2496)], 'color':[0 for _ in range(2496)]})
total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(2496)], 'x':[0 for _ in range(2496)], 'color':[0 for _ in range(2496)]})

total_df_origin_data_cnt = 0
total_df_smoothed_data_cnt = 0
baseline_cnt = 0
figure, ax = plt.subplots()
sns.set(style="whitegrid")
sns.set_palette(sns.color_palette("RdYlBu", 8))
for idx, ascend in enumerate([0, 20, 40, 60, 80, 180]):    
    data_file = "base" + str(210 + ascend)
    files = os.listdir("./{}/".format(data_file))
    print(files)

    for csv in files:
        if csv[-4:] != ".csv":
            continue
        df = pd.read_csv("./{}/{}".format(data_file, csv))
        smoothed = tensorboard_smoothing(df['Value'])
        df['smoothed'] = pd.Series(smoothed)
        df['x'] = df.index

        for i in range(416):
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = str(210 + ascend)
            total_df_smoothed_data_cnt += 1
            baseline_cnt += 1
        for i in range(416):
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = str(210 + ascend)
            total_df_origin_data_cnt += 1



    # df_baseline = pd.read_csv("../../plot/res_ascend{}.csv".format(str(ascend)))
    # for i in range(1800):
    #     total_df_smoothed_data.loc[baseline_cnt, 'data'] = df_baseline.loc[i, 'data']
    #     total_df_smoothed_data.loc[baseline_cnt, 'x'] = df_baseline.loc[i, 'x']
    #     total_df_smoothed_data.loc[baseline_cnt, 'color'] = "Baseline"
    #     baseline_cnt += 1

mark = [50 * i + 10 for i in range(0, 8)]

# palette=sns.cubehelix_palette(8)
# palette=sns.color_palette("RdYlBu", 8)
# palette=sns.hls_palette(8 , l = .5, s = .7)
# sns.color_palette(["#9CB3D4", "#ECEDFF", "#AF5A76"])
# sns.color_palette(["#DD8452", "#4C72B0", "#55A868", "#FF5E80","#66E1B1"])

sns.lineplot(x="x", y="data", data=total_df_origin_data, hue='color', palette=sns.color_palette("RdYlBu", 6), legend=False, alpha=0.2)
s = sns.lineplot(x="x", y="data", data=total_df_smoothed_data, style='color', hue='color', dashes=False, linestyle="-", 
                markers=['o', 'o', 'o', 'o'], **dict(linewidth=2, ), markersize=7, markeredgewidth=1, markevery=mark,
                palette=sns.color_palette("RdYlBu", 6),  alpha=1)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }
plt.title(model_name)
plt.grid(linestyle='-.')
plt.tick_params(labelsize=13)
# plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '300k','600k','900k', '1200k', '1500k'])
plt.xticks([0, 80, 160, 240, 320, 400], ['0', '80k','160k','240k', '320k', '400k'])
# ax.legend(loc='lower right')
plt.legend(prop=font1, loc=1, markerscale=1,)
plt.xlabel('Step', fontdict=font1)
plt.ylabel('Model Load Times', fontdict=font1)
plt.tight_layout()
plt.savefig("./{}_dis_precent_compare_400k.pdf".format(model_name))
plt.savefig("./{}_dis_precent_compare_400k.png".format(model_name))
plt.show()
print("-------------------------------")