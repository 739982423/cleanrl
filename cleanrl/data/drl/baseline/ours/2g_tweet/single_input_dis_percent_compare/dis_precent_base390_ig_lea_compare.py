# 该文件绘制base390输入流下 四个模型的 dis precent 收敛情况 与两个baseline的横线对比图，四个模型分别对应四个子图，每张子图上一条曲线，两条横线，四张子图组成一个大图
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

# 每行的第一列是ascend，第二到五列分别是resnet50_dis_percent,vgg19_dis_percent,densenet201_dis_percent,mobilenet_dis_percent
min_workload_dis_percent_matrix = [
                                    [0,0.0,0.0,0.0,0.0],
                                    [20,0.0,0.0,0.00021446257628043907,0.0],
                                    [40,0.0,0.0,0.007844395714107432,0.0],
                                    [60,0.0,0.008251332513325133,0.035194035243303605,0.0],
                                    [80,0.0,0.013452065176203107,0.05760302020063631,0.0],
                                    [180,0.05100135505692306,0.12755335714940003,0.15581273483404132,0.0]
                                  ]

igniter_dis_percent_matrix = [
                                [0,0.0,0.0035345296356715608,0.0,0.0],
                                [20,0.0,0.0,0.0,0.0],
                                [40,0.0,0.0,0.0,0.0],
                                [60,0.0,0.0,0.01767092016882626,0.0],
                                [80,0.0,0.0023841101427308324,0.051757470581967085,0.0],
                                [180,0.0407544572836977,0.0575290830814326,0.0668059225563323,0.0]
                             ]

model_name = "DenseNet201"
model_names = ['ResNet50', 'VGG19', 'DenseNet201', 'Mobilenet V2']
file_model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']


cur_file_base = "390"

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
    mark = [50 * i + 10 for i in range(0, 8)]

    # 因为只有一个base的数据，所以长度设置为一个csv文件的数据长度：416个点
    total_df_origin_data = pd.DataFrame({'data':[0 for _ in range(416)], 'x':[0 for _ in range(416)], 'color':[0 for _ in range(416)]})
    total_df_smoothed_data = pd.DataFrame({'data':[0 for _ in range(416)], 'x':[0 for _ in range(416)], 'color':[0 for _ in range(416)]})

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

        for i in range(416):
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'data'] = df.loc[i, 'smoothed']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_smoothed_data.loc[total_df_smoothed_data_cnt, 'color'] = "DRL"
            total_df_smoothed_data_cnt += 1

        for i in range(416):
            total_df_origin_data.loc[total_df_origin_data_cnt, 'data'] = df.loc[i, 'Value']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'x'] = df.loc[i, 'x']
            total_df_origin_data.loc[total_df_origin_data_cnt, 'color'] = "DRL"
            total_df_origin_data_cnt += 1

    
    ax[r][c].plot([i for i in range(len(list(total_df_origin_data['data'])))], list(total_df_origin_data['data']), color="#4C72B0", alpha=0.2)
    ax[r][c].plot([i for i in range(len(list(total_df_smoothed_data['data'])))], list(total_df_smoothed_data['data']), color="#4C72B0", marker='o', 
             markersize=5, markeredgewidth=1, markevery=mark, alpha=1, label="DRL")
    

# ------------------ 这里是要修改的baseline部分 --------------------
    ax[r][c].hlines(xmin=0, xmax=416, y=min_workload_dis_percent_matrix[5][idx + 1], label="Least Loaded", linestyles=":", colors='violet',lw=2)
    ax[r][c].hlines(xmin=0, xmax=416, y=igniter_dis_percent_matrix[5][idx + 1], label="IGniter+", linestyles="--", colors='k',lw=2)
# -------------------------------------------------------------------


    
    ax[r][c].set_xticks([0, 80, 160, 240, 320, 400], ['0', '80k','160k','240k', '320k', '400k'])
    ax[r][c].grid(linestyle='-.')
    ax[r][c].tick_params(labelsize=13)
    loc = 1
    if r == 0 and c == 0:
        loc = 7
        ax[r][c].set_ylim([0,0.062])
    if r == 0 and c == 1:
        ax[r][c].set_ylim([0,0.275])
    ax[r][c].legend(prop=font1, loc=loc, markerscale=1,)
    
    ax[r][c].set_title(model_names[idx], fontdict=font2)
    ax[r][c].set_xlabel('Step', fontdict=font1)
    ax[r][c].set_ylabel('SLO Violation', fontdict=font1)
    # plt.tight_layout()









plt.savefig("./base390_{}_dis_precent_ig_le_compare_400k.pdf".format(model_name))
plt.savefig("./base390_{}_dis_precent_ig_le_compare_400k.png".format(model_name))
plt.show()
print("-------------------------------")
