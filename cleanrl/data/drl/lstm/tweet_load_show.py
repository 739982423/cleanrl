import matplotlib.pyplot as plt
import csv
import scipy

# 1.生成数据集

def get_input(load_data_length=5000):
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
            input_stream.append(int(row[1]) // 40)

    return input_stream

model_names = ['resnet50', 'vgg19', 'densenet201', 'mobilenet', 'alexnet', 'inception']  

input_stream = get_input(3000)
input_stream = scipy.signal.savgol_filter(input_stream,15,3)
for idx, num in enumerate(input_stream):
    input_stream[idx] = int(num)


n = 300
print(input_stream[:n])

densenet201_input_stream = list(input_stream[:n]) + [0,0,0,0,0]
mobilenet_input_stream = list(input_stream[n:2*n]) + [0,0,0,0,0]
resnet50_input_stream = list(input_stream[2*n:3*n]) + [0,0,0,0,0]
alexnet_input_stream = list(input_stream[3*n:4*n]) + [0,0,0,0,0]
inception_input_stream = list(input_stream[4*n:5*n]) + [0,0,0,0,0]
vgg19_input_stream = list(input_stream[5*n:6*n]) + [0,0,0,0,0]

# for i in range(0, 6 * n):
#     input_stream[i] += 30

# densenet201_input_stream += list(input_stream[:n]) + [0,0,0,0,0]
# mobilenet_input_stream += list(input_stream[n:2*n]) + [0,0,0,0,0]
# resnet50_input_stream += list(input_stream[2*n:3*n]) + [0,0,0,0,0]
# alexnet_input_stream += list(input_stream[3*n:4*n]) + [0,0,0,0,0]
# inception_input_stream += list(input_stream[4*n:5*n]) + [0,0,0,0,0]
# vgg19_input_stream += list(input_stream[5*n:6*n]) + [0,0,0,0,0]

# for i in range(0, 6 * n):
#     input_stream[i] += 30

# densenet201_input_stream += list(input_stream[:n]) + [0,0,0,0,0]
# mobilenet_input_stream += list(input_stream[n:2*n]) + [0,0,0,0,0]
# resnet50_input_stream += list(input_stream[2*n:3*n]) + [0,0,0,0,0]
# alexnet_input_stream += list(input_stream[3*n:4*n]) + [0,0,0,0,0]
# inception_input_stream += list(input_stream[4*n:5*n]) + [0,0,0,0,0]
# vgg19_input_stream += list(input_stream[5*n:6*n]) + [0,0,0,0,0]  

x = [i for i in range(len(densenet201_input_stream))]
plt.subplot(2,3,1)
plt.plot(x, densenet201_input_stream, label="densenet201")
plt.legend()
plt.subplot(2,3,2)
plt.plot(x, mobilenet_input_stream, label="mobilenet")
plt.legend()
plt.subplot(2,3,3)
plt.plot(x, resnet50_input_stream, label="resnet50")
plt.legend()
plt.subplot(2,3,4)
plt.plot(x, alexnet_input_stream, label="alexnet")
plt.legend()
plt.subplot(2,3,5)
plt.plot(x, inception_input_stream, label="inception")
plt.legend()
plt.subplot(2,3,6)
plt.plot(x, vgg19_input_stream, label="vgg19")
plt.legend()
plt.show()

with open("tweet_load_base300.csv", mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['densenet201'] + densenet201_input_stream)
    writer.writerow(['mobilenet'] + mobilenet_input_stream)
    writer.writerow(['resnet50'] + resnet50_input_stream)
    writer.writerow(['alexnet'] + alexnet_input_stream)
    writer.writerow(['inception'] + inception_input_stream)
    writer.writerow(['vgg19'] + vgg19_input_stream)