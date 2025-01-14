import csv
import matplotlib.pyplot as plt
import collections
import numpy as np

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

input_stream = getTweetInput(-50)

ours_dis_a002 = collections.defaultdict(list)
ours_dis_a0002 = collections.defaultdict(list)
igniter_dis = collections.defaultdict(list)
leastload_dis = collections.defaultdict(list)


# 获取dis nums
figure, ax = plt.subplots(4,1)

font1 = {'family': 'STZhongsong',
            'weight': 'normal',
            'size': 17,
            }
font2 = {'family': 'STZhongsong',
            'weight': 'normal',
            'size': 14,
            }

for i in range(4):
    labels = ax[i].get_xticklabels() + ax[i].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

idx = 0
for method in ['leastload','igniter', 'ours', 'ours']:
    print(idx)
    file_name = "./{}/{}/plotdata/dis_nums.csv".format(method, "base290")
    if idx == 2:
        file_name = "./{}/{}/plotdata/dis_nums.csv".format(method, "alpha_0.02_base290")
    with open(file_name, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            if method == 'igniter':
                igniter_dis['resnet50'].append(float(row[0]))
                igniter_dis['vgg19'].append(float(row[1]))
                igniter_dis['densenet201'].append(float(row[2]))
                igniter_dis['mobilenet'].append(float(row[3]))
            elif method == 'leastload':
                leastload_dis['resnet50'].append(float(row[0]))
                leastload_dis['vgg19'].append(float(row[1]))
                leastload_dis['densenet201'].append(float(row[2]))
                leastload_dis['mobilenet'].append(float(row[3]))
            elif method == 'ours' and idx == 3:
                ours_dis_a0002['resnet50'].append(float(row[0]))
                ours_dis_a0002['vgg19'].append(float(row[1]))
                ours_dis_a0002['densenet201'].append(float(row[2]))
                ours_dis_a0002['mobilenet'].append(float(row[3]))
            elif method == 'ours' and idx == 2:
                ours_dis_a002['resnet50'].append(max(0, float(row[0])-10))
                ours_dis_a002['vgg19'].append(max(0, float(row[1])-10))
                ours_dis_a002['densenet201'].append(float(row[2]))
                ours_dis_a002['mobilenet'].append(float(row[3]))

    if method == "igniter":
        ax[idx].set_title("IGniter", fontdict=font1)
    elif method == "leastload":
        ax[idx].set_title("LeastLoad", fontdict=font1)
    elif method == "ours" and idx == 3:
        ax[idx].set_title("DRL+LSTM β=0.002", fontdict=font1)
    elif method == "ours" and idx == 2:
        ax[idx].set_title("DRL+LSTM β=0.02", fontdict=font1)

    total_width, n = 1, 4
    width = total_width / n
    x = np.arange(240)
    x = x - (total_width - width) / 2
    for i in range(4):
        model_name = model_names[i]
        if method == 'igniter':
            y = igniter_dis[model_name]
        elif method == 'leastload':
            y = leastload_dis[model_name]
        elif method == 'ours' and idx == 3:
            y = ours_dis_a0002[model_name]
        elif method == 'ours' and idx == 2:
            y = ours_dis_a002[model_name]

        if model_name == 'resnet50':
            model_name = 'ResNet50'
        elif model_name == 'vgg19':
            model_name = 'VGG19'
        elif model_name == 'densenet201':
            model_name = 'DenseNet201'
        elif model_name == 'mobilenet':
            model_name = 'MobileNet_V2'
        ax[idx].bar(x + width * i, y, width=width, label=model_name)
        ax[idx].set_xticks([30,60,90,120,150,180,210,240])
        ax[idx].tick_params(axis='x', labelsize=16)
        ax[idx].tick_params(axis='y', labelsize=16)
        ax[idx].grid(linestyle='-.')
        ax[idx].set_xlabel('时间（秒）', fontdict=font1)
        ax[idx].set_ylabel('请求丢弃数量', fontdict=font1)
        ax[idx].legend(prop=font2, loc=2)
    idx += 1
# plt.show()

# ours_dis_a002 = collections.defaultdict(list)
# ours_dis_a0002 = collections.defaultdict(list)
# igniter_dis = collections.defaultdict(list)
# leastload_dis = collections.defaultdict(list)

dis_nums = collections.defaultdict(int)
for k, v in ours_dis_a0002.items():
    dis_nums[k] += sum(v)
print("dis percent, resnet50:", dis_nums['resnet50'], dis_nums['resnet50'] / sum(input_stream['resnet50']))
print("dis percent, vgg19:", dis_nums['vgg19'], dis_nums['vgg19'] / sum(input_stream['vgg19']))
print("dis percent, densenet201:", dis_nums['densenet201'], dis_nums['densenet201'] / sum(input_stream['densenet201']))
print("dis percent, mobilenet:", dis_nums['mobilenet'], dis_nums['mobilenet'] / sum(input_stream['mobilenet']))
print("dis percent, total:", 
      (dis_nums['resnet50'] + dis_nums['vgg19'] + dis_nums['densenet201'] + dis_nums['mobilenet'])\
      / (sum(input_stream['resnet50']) + sum(input_stream['vgg19']) + sum(input_stream['densenet201']) + sum(input_stream['mobilenet'])))
