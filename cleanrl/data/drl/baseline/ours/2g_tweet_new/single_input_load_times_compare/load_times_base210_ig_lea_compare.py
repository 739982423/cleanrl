# 该文件绘制base290输入流下 四个模型的 dis precent 收敛情况 与两个baseline的横线对比图，四个模型分别对应四个子图，每张子图上一条曲线，两条横线，四张子图组成一个大图
# 并放置一个总的legend
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

# 每行的第一列是ascend，第二到五列分别是resnet50_load_times,vgg19_load_times,densenet201_load_times,mobilenet_load_times
min_workload_load_times_matrix = [
                                    [0,40,34,42,33],
                                    [20,55,56,46,66],
                                    [40,62,46,26,53],
                                    [60,60,48,10,62],
                                    [80,27,18,11,28],
                                    [180,4,5,4,6]
                                  ]

igniter_load_times_matrix = [
                                [0,5,5,8,1],
                                [20,1,1,15,1],
                                [40,1,1,8,1],
                                [60,4,3,3,4],
                                [80,7,7,3,3],
                                [180,2,2,2,3]
                             ]

model_name = "DenseNet201"
model_names = ['ResNet50', 'VGG19', 'DenseNet201', 'Mobilenet V2']
file_model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']


cur_file_base = "210"

figure, ax = plt.subplots(2,2)

font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12,
            }
font2 = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 13,
            }

for i in range(2):
    for j in range(2):
        labels = ax[i][j].get_xticklabels() + ax[i][j].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

for idx, m in enumerate(file_model_names):
    r, c = 0, 0
    if idx == 1:
        r = 0
        c = 1
    elif idx == 2:
        r = 1
        c = 0
    elif idx == 3:
        r = 1
        c = 1

    sns.set(style="whitegrid")
    # pattle = sns.color_palette("RdYlBu", 1) 
    mark = [18 * i + 12 for i in range(0, 11)]

    # 因为只有一个base的数据，所以长度设置为一个csv文件的数据长度：416个点
    total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(208)], 'x':[0 for _ in range(208)], 'color':[0 for _ in range(208)]})
    total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(208)], 'x':[0 for _ in range(208)], 'color':[0 for _ in range(208)]})

    total_df_origin_data_cnt = 0
    total_df_smoothed_data_cnt = 0


    data_file = "base" + cur_file_base
    path = "./{}/{}/".format(m, data_file)
    files = os.listdir(path)
    print(path, files)

    for csv in files:
        if csv[-4:] != ".csv":
            continue
        df = pd.read_csv("./{}/{}/{}".format(m, data_file, csv))
        smoothed = tensorboard_smoothing(df['Value'])
        df['smoothed'] = pd.Series(smoothed)
        df['x'] = df.index

        for i in range(208):
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = "DRL"
            total_df_smoothed_data_cnt += 1

        for i in range(208):
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = "DRL"
            total_df_origin_data_cnt += 1

    
    ax[r][c].plot([i for i in range(len(list(total_df_origin_data['data'])))], list(total_df_origin_data['data']), color="#4C72B0", alpha=0.2)
    ax[r][c].plot([i for i in range(len(list(total_df_smoothed_data['data'])))], list(total_df_smoothed_data['data']), color="#4C72B0", marker='o', 
             markersize=5, markeredgewidth=1, markevery=mark, alpha=1, label="DRL")
    

# ------------------ 这里是要修改的baseline部分 --------------------
    ax[r][c].hlines(xmin=0, xmax=208, y=min_workload_load_times_matrix[0][idx + 1], label="Least Loaded", linestyles=":", colors='violet',lw=2)
    ax[r][c].hlines(xmin=0, xmax=208, y=igniter_load_times_matrix[0][idx + 1], label="IGniter+", linestyles="--", colors='k',lw=2)
# -------------------------------------------------------------------


    
    ax[r][c].set_xticks([0, 40, 80, 120, 160, 200], ['0', '80k','160k','240k', '320k', '400k'])
    ax[r][c].grid(linestyle='-.')
    ax[r][c].tick_params(labelsize=13)
    loc = 1
    ax[r][c].legend(prop=font1, loc=loc, markerscale=1,)

    ax[r][c].set_title(model_names[idx], fontdict=font2)
    ax[r][c].set_xlabel('Step', fontdict=font1)
    ax[r][c].set_ylabel('Model Loading Times', fontdict=font1)
    # plt.tight_layout()









plt.savefig("./base210_{}_load_times_ig_le_compare_400k.pdf".format(model_name))
plt.savefig("./base210_{}_load_times_ig_le_compare_400k.png".format(model_name))
plt.show()
print("-------------------------------")
