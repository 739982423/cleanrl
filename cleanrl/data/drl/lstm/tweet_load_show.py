import matplotlib.pyplot as plt
import csv
import scipy

# 1.生成数据集

def get_input(load_data_length=2880):
    input_stream = []
    with open("./tweet_load.csv", mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            line += 1
            if int(row[1]) > 15000:
                row[1] = 15000
            if line - 2 == load_data_length:
                break
            input_stream.append(int(row[1]))
            # input_stream.append(int(row[1]))

    return input_stream

model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet']  

input_stream = get_input(2880)

figure, ax = plt.subplots()
plt.plot([i for i in range(len(input_stream[:1440]))], input_stream[:1440], label="Twitter追踪")
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

font2 = {'family': 'STZhongsong',
            'weight': 'normal',
            'size': 13,
            }
plt.grid(linestyle="-.")
plt.xlabel("时间（分钟）", fontdict=font2)
plt.ylabel("请求数", fontdict=font2)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(prop=font2, loc=1, markerscale=1,)
plt.show()

# input_stream = scipy.signal.savgol_filter(input_stream,15,3)

# input_stream = input_stream[600:]
# for idx, num in enumerate(input_stream):
#     input_stream[idx] = int(num) + 40


# n = 240
# # print(input_stream[:n])
# # plt.plot([i for i in range(len(input_stream))], input_stream)
# # plt.show()


# densenet201_input_stream = list(input_stream[:n])
# vgg19_input_stream = list(input_stream[n:2*n])
# resnet50_input_stream = list(input_stream[2*n:3*n])
# alexnet_input_stream = list(input_stream[3*n:4*n])
# inception_input_stream = list(input_stream[4*n:5*n])
# mobilenet_input_stream = list(input_stream[5*n:6*n])


# densenet201_input_stream2 = []
# mobilenet_input_stream2 = []
# resnet50_input_stream2 = []
# vgg19_input_stream2 = []
# for i in range(len(densenet201_input_stream)):
#     if i % 1 == 0:
#         densenet201_input_stream2.append(densenet201_input_stream[i])
#         mobilenet_input_stream2.append(mobilenet_input_stream[i])
#         resnet50_input_stream2.append(resnet50_input_stream[i])
#         vgg19_input_stream2.append(vgg19_input_stream[i])




# figure, ax = plt.subplots()
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # x = [i for i in range(len(densenet201_input_stream))]
# x = [i for i in range(len(densenet201_input_stream2))]
# plt.subplot(2,2,1)
# # plt.plot(x, densenet201_input_stream, label="DenseNet201")
# plt.plot(x, densenet201_input_stream2, label="DenseNet201")
# plt.grid(linestyle="-.")
# plt.xlabel("Time(s)", fontdict=font1)
# plt.ylabel("Request Number", fontdict=font1)
# plt.legend(prop=font1, loc=1, markerscale=1,)

# plt.subplot(2,2,2)
# # plt.plot(x, mobilenet_input_stream, label="MobileNet V2")
# plt.plot(x, mobilenet_input_stream2, label="MobileNet V2")
# plt.grid(linestyle="-.")
# plt.xlabel("Time(s)", fontdict=font1)
# plt.ylabel("Request Number", fontdict=font1)
# plt.legend(prop=font1, loc=1, markerscale=1,)

# plt.subplot(2,2,3)
# # plt.plot(x, resnet50_input_stream, label="ResNet50")
# plt.plot(x, resnet50_input_stream2, label="ResNet50")
# plt.grid(linestyle="-.")
# plt.xlabel("Time(s)", fontdict=font1)
# plt.ylabel("Request Number", fontdict=font1)
# plt.legend(prop=font1, loc=1, markerscale=1,)

# plt.subplot(2,2,4)
# # plt.plot(x, vgg19_input_stream, label="VGG19")
# plt.plot(x, vgg19_input_stream2, label="VGG19")
# plt.grid(linestyle="-.")
# plt.xlabel("Time(s)", fontdict=font1)
# plt.ylabel("Request Number", fontdict=font1)
# plt.legend(prop=font1, loc=1, markerscale=1,)
# # plt.subplot(2,3,5)
# # plt.plot(x, alexnet_input_stream, label="alexnet")
# # plt.legend()
# # plt.subplot(2,3,6)
# # plt.plot(x, inception_input_stream, label="inception")
# # plt.legend()
# plt.show()

# with open("tweet_load_base240.csv", mode="w", encoding="utf-8", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['densenet201'] + alexnet_input_stream)
#     writer.writerow(['mobilenet'] + mobilenet_input_stream)
#     writer.writerow(['resnet50'] + resnet50_input_stream)
#     writer.writerow(['vgg19'] + vgg19_input_stream)