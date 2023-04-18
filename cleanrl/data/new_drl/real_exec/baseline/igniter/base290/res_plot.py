import csv
import os
import matplotlib.pyplot as plt
import collections

root_file = "./get_res/res/"
res_file = root_file + "/origin_data"


time_duration = 240

model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']  
# model_names = ['densenet201']  

input_stream_file = "/home/hpj/project/cleanrl/cleanrl/data/drl/lstm/tweet_load_base240.csv"

input_stream = collections.defaultdict(list)
input_ascend = 80


with open(input_stream_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        input_stream[row[0]] = []
        for num in row[1:]:
            input_stream[row[0]].append(int(num) + input_ascend)
    
plt.figure()
for idx, model_name in enumerate(model_names):

    throughput_list = [[0, 0] for _ in range(time_duration)]
    ratio_list = [[0, 0] for _ in range(time_duration)]
    ratio_throughput_list = [[0, 0] for _ in range(time_duration)]

    for i in range(time_duration):
        data_dir = res_file + "/" + str(i)
        files = os.listdir(data_dir)
        for csv_file in files:
            if csv_file [-4:] != ".csv":
                continue
            # print(csv)
            print(data_dir + "/" + csv_file)
            with open(data_dir + "/" + csv_file, mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == model_name:
                        if csv_file == "gpu0.csv":
                            throughput_list[i][0] = float(row[3])
                            ratio_throughput_list[i][0] = float(row[4])
                        elif csv_file == "gpu1.csv":
                            throughput_list[i][1] = float(row[3])
                            ratio_throughput_list[i][1] = float(row[4])
    plt.subplot(4,1,idx + 1)
    plt.title(model_name)
    if model_name == "mobilenet":
        GPU0_throughput = [ratio_throughput_list[i][0] / 2 for i in range(time_duration)]
        GPU1_throughput = [ratio_throughput_list[i][1] / 2 for i in range(time_duration)]
    elif model_name == "resnet50":
        GPU0_throughput = [ratio_throughput_list[i][0] / 1.25 for i in range(time_duration)]
        GPU1_throughput = [ratio_throughput_list[i][1] / 1.25  for i in range(time_duration)]
    else:
        GPU0_throughput = [ratio_throughput_list[i][0] * 1.1 for i in range(time_duration)]
        GPU1_throughput = [ratio_throughput_list[i][1] * 1.1 for i in range(time_duration)]

    GPU0_plus_GPU1_throughput = [(GPU0_throughput[i] + GPU1_throughput[i]) for i in range(time_duration)]
    x = [i for i in range(time_duration)]

    # plt.bar(x, GPU0_throughput, label="GPU0")
    # plt.bar(x, GPU1_throughput, bottom=GPU0_throughput, label="GPU1")
    
    plt.plot(x, GPU0_throughput, label="GPU0")
    plt.fill_between(x=x, y1=0, y2=GPU0_throughput, alpha=0.5)

    plt.plot(x, GPU0_plus_GPU1_throughput, label="GPU1")
    plt.fill_between(x=x, y1=GPU0_throughput, y2=GPU0_plus_GPU1_throughput, alpha=0.5)

    # plt.plot([i for i in range(time_duration)], [throughput_list[i][0] + throughput_list[i][1] for i in range(time_duration)], label="SUM")
    plt.plot([i for i in range(time_duration)], input_stream[model_name][:time_duration], label="Input")
    plt.legend()
    # plt.savefig("{}_{}.png".format("base290", model_name))
    # plt.show()

    if not os.path.exists("{}/summary_data".format(root_file)):
        os.mkdir("{}/summary_data".format(root_file))
   
    with open(root_file + "/summary_data/{}.csv".format(model_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for i in range(time_duration):
            gpu0_throughput = throughput_list[i][0]
            gpu1_throughput = throughput_list[i][1]
            ratio_gpu0_throughput = ratio_throughput_list[i][0]
            ratio_gpu1_throughput = ratio_throughput_list[i][1]

            writer.writerow([gpu0_throughput, gpu1_throughput, ratio_gpu0_throughput, ratio_gpu1_throughput])
plt.savefig("overall.png")
plt.show()
