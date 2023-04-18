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

input_stream = getTweetInput(80)
ours_throughput = collections.defaultdict(list)
igniter_throughput = collections.defaultdict(list)
leastload_throughput = collections.defaultdict(list)

ours_dis = collections.defaultdict(list)
igniter_dis = collections.defaultdict(list)
leastload_dis = collections.defaultdict(list)


# plt.figure()
for idx, model_name in enumerate(model_names):
    plt.subplot(4,1,idx + 1)
    x = [i for i in range(len(input_stream[model_name]))]
    
    for method in ['igniter', 'leastload', 'ours']:
        tmp_throughput = []
        print("./{}/{}/plotdata/summary_data/{}.csv".format(method, "base290", model_name))
        with open("./{}/{}/plotdata/summary_data/{}.csv".format(method, "base290", model_name), mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
                ratio_gpu1_throughput = float(row[2])
                ratio_gpu2_throughput = float(row[3])
                tmp_throughput.append(ratio_gpu1_throughput + ratio_gpu2_throughput)
        if method == 'igniter':
            igniter_throughput[model_name] = tmp_throughput
        elif method == 'leastload':
            leastload_throughput[model_name] = tmp_throughput
        elif method == 'ours':
            ours_throughput[model_name] = tmp_throughput

    # plt.title(model_name)
    # plt.plot(x, input_stream[model_name], label="Input")
    # plt.plot(x, leastload_throughput[model_name], label="Least Load")
    # plt.plot(x, igniter_throughput[model_name], label="IGniter")
    # plt.plot(x, ours_throughput[model_name], label="DRL+LSTM")

    total_width, n = 0.5, 3
    width = total_width / n
    x = np.arange(120)
    x = x - (total_width - width) / 2
    plt.plot(x, input_stream[model_name][120:240], label="Input")
    plt.bar(x, leastload_throughput[model_name][120:240], width=width, label="Least Load")
    plt.bar(x + width, igniter_throughput[model_name][120:240], width=width, label="IGniter")
    plt.bar(x + 2 * width, ours_throughput[model_name][120:240], width=width, label="DRL+LSTM")
    if idx == 0:
        plt.legend()
plt.show()
