import csv
import os

model_kernel_data = dict()
summary_file = "../../summary3.csv"
with open(summary_file, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line < 1:
            line += 1
            continue 
        key = row[0] + "_b" + row[1] + "_g" + row[2]
        model_kernel_data[key] = row

target_file = "cleaned_d_r_v.csv"

write_lines = [["model1 name","gpu1","batch1","total latency","exec times", "avg latency",
                "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency", "",
                "model2 name","gpu2","batch2","total latency","exec times", "avg latency",
                "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency", "",
                "model3 name","gpu3","batch3","total latency","exec times", "avg latency",
                "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency",]]

with open(target_file, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    group_cnt = 0
    idx = 1
    for row in reader:
        print(row)
        if idx % 4 == 1:
            group_data1 = []
            group_data2 = []
            group_data3 = []
            group_data4 = []
            idx += 1
            continue
        elif idx % 4 == 2:
            # 基础数据获取
            m_name1 = row[0]
            gpu1 = row[1]
            if gpu1 == '25':
                gpu1 = '20'
            if gpu1 == '75':
                gpu1 = '70'
            batch1 = row[2]
            total_latency1 = row[3]
            exec_times1 = row[4]
            avg_latency1 = row[5]
            group_data1 = [m_name1, gpu1, batch1, total_latency1, exec_times1, avg_latency1]

            # 核函数相关数据获取
            key = m_name1 + "_b" + batch1 + "_g" + gpu1
            kernel_data1 = model_kernel_data[key]
            # 从第四个数据开始append，因为前三个是模型名、批量、GPU资源量
            for i in range(3, len(kernel_data1)):
                group_data1.append(kernel_data1[i])
            idx += 1

        elif idx % 4 == 3:
            # 基础数据获取
            m_name2 = row[0]
            gpu2 = row[1]
            if gpu2 == '25':
                gpu2 = '20'
            if gpu2 == '75':
                gpu2 = '70'
            batch2 = row[2]
            total_latency2 = row[3]
            exec_times2 = row[4]
            avg_latency2 = row[5]
            group_data2 = [m_name2, gpu2, batch2, total_latency2, exec_times2, avg_latency2]

            # 核函数相关数据获取
            key = m_name2 + "_b" + batch2 + "_g" + gpu2
            kernel_data2 = model_kernel_data[key]
            # 从第四个数据开始append，因为前三个是模型名、批量、GPU资源量
            for i in range(3, len(kernel_data2)):
                group_data2.append(kernel_data2[i])
            idx += 1

        elif idx % 4 == 0:
            # 基础数据获取
            m_name3 = row[0]
            gpu3 = row[1]
            if gpu3 == '25':
                gpu3 = '20'
            if gpu3 == '75':
                gpu3 = '70'
            batch3 = row[2]
            total_latency3 = row[3]
            exec_times3 = row[4]
            avg_latency3 = row[5]
            group_data3 = [m_name3, gpu3, batch3, total_latency3, exec_times3, avg_latency3]

            # 核函数相关数据获取
            key = m_name3 + "_b" + batch3 + "_g" + gpu3
            kernel_data3 = model_kernel_data[key]
            # 从第四个数据开始append，因为前三个是模型名、批量、GPU资源量
            for i in range(3, len(kernel_data3)):
                group_data3.append(kernel_data3[i])
            idx += 1

            write_line = group_data1 + [""] + group_data2 + [""] + group_data3 + [""] +\
                         group_data4
            write_lines.append(write_line)

# 写结果部分
res_file = "./pro_" + target_file
with open(res_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for row in write_lines:
        writer.writerow(row)
