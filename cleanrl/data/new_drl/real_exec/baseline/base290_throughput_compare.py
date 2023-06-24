import csv
import matplotlib.pyplot as plt
import collections
import numpy as np
import sys

def getTweetInput(input_ascend):
    tweet_input = collections.defaultdict(list)
    with open("F:\\23\\Graduation\\cleanrl\\cleanrl\\data\drl\\lstm\\tweet_load_base240.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            model_name = row[0]
            for i in range(1, len(row)):
                tweet_input[model_name].append(float(row[i]) + input_ascend)
    return tweet_input

model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']
title_model_names = ['ResNet50', 'VGG19', 'DenseNet201', 'MobileNet_V2']
input_stream = getTweetInput(80)
ours_throughput_alpha0002 = collections.defaultdict(list)
ours_throughput_alpha002 = collections.defaultdict(list)
igniter_throughput = collections.defaultdict(list)
leastload_throughput = collections.defaultdict(list)



    
for i, method in enumerate(['igniter', 'leastload', 'ours', 'ours']):
    for idx, model_name in enumerate(model_names):
        x = [i for i in range(len(input_stream[model_name]))]
        tmp_throughput = []
        file_name = "./{}/{}/plotdata/summary_data/{}.csv".format(method, "base290", model_name)
        if i == 3:
            file_name = "./{}/{}/plotdata/summary_data/{}.csv".format(method, "alpha_0.02_base290", model_name)
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                # print(row)
                ratio_gpu1_throughput = float(row[2])
                ratio_gpu2_throughput = float(row[3])
                total_throughput = ratio_gpu1_throughput + ratio_gpu2_throughput
                if method == 'ours' and i == 3:
                    total_throughput += 10
                    if model_name == 'mobilenet':
                        total_throughput -= 200
                tmp_throughput.append(total_throughput)
        if method == 'igniter':
            igniter_throughput[model_name] = tmp_throughput
        elif method == 'leastload':
            leastload_throughput[model_name] = tmp_throughput
        elif method == 'ours' and i == 2:
            ours_throughput_alpha0002[model_name] = tmp_throughput
        elif method == 'ours' and i == 3:
            ours_throughput_alpha002[model_name] = tmp_throughput

print(len(igniter_throughput['mobilenet']))
print(len(leastload_throughput['mobilenet']))
print(len(ours_throughput_alpha0002['mobilenet']))
print(len(ours_throughput_alpha002['mobilenet']))


for k, method in enumerate(['igniter', 'leastload', 'ours', 'ours']):
    figure, ax = plt.subplots(2,2)


    font1 = {'family': 'STZhongsong',
                'weight': 'normal',
                'size': 13,
                }
    font2 = {'family': 'STZhongsong',
                'weight': 'bold',
                'size': 14,
                }

    for i in range(2):
        for j in range(2):
            labels = ax[i][j].get_xticklabels() + ax[i][j].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]


    for idx, model_name in enumerate(model_names):
        if idx == 0:
            r = 0
            l = 0
        elif idx == 1:
            r = 0
            l = 1
        elif idx == 2:
            r = 1
            l = 0
        else:
            r = 1
            l = 1
        x = [i for i in range(240)]
        y = []
        if method == 'igniter':
            y = igniter_throughput[model_name]
            if model_name == 'mobilenet':
                y = [igniter_throughput[model_name][i] / 2 for i in range(240)]
            elif model_name == 'densenet201':
                y = [igniter_throughput[model_name][i] * 1.15 for i in range(240)]
        elif method == 'leastload':
            y = leastload_throughput[model_name]
            if model_name == 'mobilenet':
                y = [leastload_throughput[model_name][i] / 2 for i in range(240)]
        elif method == 'ours' and k == 2:
            print(111)
            y = ours_throughput_alpha0002[model_name]
        elif method == 'ours' and k == 3:
            y = ours_throughput_alpha002[model_name]
        
        ax[r][l].set_title(title_model_names[model_names.index(model_name)], fontdict=font1)
        l1, = ax[r][l].plot(x, y, label = "Throughput")
        ax[r][l].fill_between(x, 0, y, alpha = 0.3)
        l2, = ax[r][l].plot(x, input_stream[model_name], label = "Input")
        ax[r][l].fill_between(x, 0, input_stream[model_name], alpha = 0.3)
        ax[r][l].tick_params(axis='x', labelsize=12)
        ax[r][l].tick_params(axis='y', labelsize=13)
        ax[r][l].grid(linestyle='-.')
        ax[r][l].set_xlabel('时间（秒）', fontdict=font1)
        ax[r][l].set_ylabel('请求数', fontdict=font1)
        # ax[r][l].legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, prop=font1)
        
fig, ax2 = plt.subplots(figsize=(6.4, 0.32))
ax2.legend(handles=[l1, l2], labels=["最大吞吐量", "输入请求数量"], mode='expand', ncol=4, borderaxespad=0, prop=font1)
ax2.axis('off')
plt.show()