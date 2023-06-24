import csv
import matplotlib.pyplot as plt
import collections
import numpy as np



a0002_base290_gpu0 = []
a0002_base290_gpu1 = []
a002_base290_gpu0 = []
a002_base290_gpu1 = []
with open("./ours/base290/plotdata/memory.csv", mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        a0002_base290_gpu0.append(float(row[0]))
        a0002_base290_gpu1.append(float(row[1]))

with open("./ours/alpha_0.02_base290/plotdata/memory.csv", mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        a002_base290_gpu0.append(float(row[0]))
        a002_base290_gpu1.append(float(row[1]))


figure, ax = plt.subplots(2,1)

font1 = {'family': 'STZhongsong',
            'weight': 'normal',
            'size': 13,
            }
font2 = {'family': 'STZhongsong',
            'weight': 'bold',
            'size': 14,
            }

for i in range(2):
    labels = ax[i].get_xticklabels() + ax[i].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


total_width, n = 1, 4
width = total_width / n
x = np.arange(240)
x = x - (total_width - width) / 2

for idx in range(2):
    x = [i for i in range(240)]
    # if idx == 0:
    #     ax[idx].plot(x, base210_memory_gpu0, label="GPU0")
    #     ax[idx].fill_between(x, 0, base210_memory_gpu0, alpha = 0.2)
        
    #     sum_throughput = [base210_memory_gpu0[i] + base210_memory_gpu1[i] for i in range(240)]
    #     ax[idx].plot(x, sum_throughput, label="GPU1")
    #     ax[idx].fill_between(x, base210_memory_gpu0, sum_throughput, alpha = 0.2)

    # elif idx == 1:
    #     ax[idx].plot(x, base290_memory_gpu0, label="GPU0")
    #     ax[idx].fill_between(x, 0, base290_memory_gpu0, alpha = 0.2)
    #     sum_throughput = [base290_memory_gpu0[i] + base290_memory_gpu1[i] for i in range(240)]
    #     ax[idx].plot(x, sum_throughput, label="GPU1")
    #     ax[idx].fill_between(x, base290_memory_gpu0, sum_throughput, alpha = 0.2)

    if idx == 1:
        ax[idx].plot(x, a0002_base290_gpu0, label="GPU0")
        ax[idx].fill_between(x, 0, a0002_base290_gpu0, alpha = 0.2)

        ax[idx].plot(x, a0002_base290_gpu1, label="GPU1")
        ax[idx].fill_between(x, 0, a0002_base290_gpu1, alpha = 0.2)
        ax[idx].set_title("β=0.002", fontdict=font1)

    elif idx == 0:
        ax[idx].plot(x, a002_base290_gpu0, label="GPU0")
        ax[idx].fill_between(x, 0, a002_base290_gpu0, alpha = 0.2)

        ax[idx].plot(x, a002_base290_gpu1, label="GPU1")
        ax[idx].fill_between(x, 0, a002_base290_gpu1, alpha = 0.2)
        ax[idx].set_title("β=0.02", fontdict=font1)

    # if idx == 0:
    #     ax[idx].bar(x, base210_memory_gpu0, width=width, label="GPU0")
    #     ax[idx].bar(x + width, base210_memory_gpu1, width=width, label="GPU1")
    # elif idx == 1:
    #     ax[idx].bar(x, base290_memory_gpu0, width=width, label="GPU0")
    #     ax[idx].bar(x + width, base290_memory_gpu1, width=width, label="GPU1")

    ax[idx].tick_params(axis='x', labelsize=12)
    ax[idx].tick_params(axis='y', labelsize=13)
    ax[idx].grid(linestyle='-.')
    ax[idx].set_xlabel('时间（秒）', fontdict=font1)
    ax[idx].set_ylabel('内存释放量（MB）', fontdict=font1)
    ax[idx].legend(prop=font1, loc=1)

plt.show()

print("a0002,g0", sum(a0002_base290_gpu0)/len(a0002_base290_gpu0))
print("a0002,g1", sum(a0002_base290_gpu1)/len(a0002_base290_gpu0))
print("a002,g0", sum(a002_base290_gpu0)/len(a0002_base290_gpu0))
print("a002,g0", sum(a002_base290_gpu1)/len(a0002_base290_gpu0))
