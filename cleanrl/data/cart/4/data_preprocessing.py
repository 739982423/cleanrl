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

target_file = "cleaned_d_m_r_v.csv"
# target_file = "cleaned_d_r_v.csv"

write_lines = [["model1 name","gpu1","batch1","total latency","exec times", "avg latency",
                "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency", "",
                "model2 name","gpu2","batch2","total latency","exec times", "avg latency",
                "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency"]]

with open(target_file, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    group_cnt = 0
    idx = 1
    for row in reader:
        if idx % 5 == 1:
            group_data1 = []
            group_data2 = []
            group_data3 = []
            group_data4 = []
            idx += 1
            continue
        elif idx % 5 == 2:
            # 基础数据获取
            m_name1 = row[0]
            gpu1 = row[1]
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

        elif idx % 5 == 3:
            # 基础数据获取
            m_name2 = row[0]
            gpu2 = row[1]
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

        elif idx % 5 == 3:
            # 基础数据获取
            m_name3 = row[0]
            gpu3 = row[1]
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

        elif idx % 5 == 4:
            # 基础数据获取
            m_name4 = row[0]
            gpu4 = row[1]
            batch4 = row[2]
            total_latency4 = row[3]
            exec_times4 = row[4]
            avg_latency4 = row[5]
            group_data4 = [m_name4, gpu4, batch4, total_latency4, exec_times4, avg_latency4]

            # 核函数相关数据获取
            key = m_name4 + "_b" + batch4 + "_g" + gpu4
            kernel_data4 = model_kernel_data[key]
            # 从第四个数据开始append，因为前三个是模型名、批量、GPU资源量
            for i in range(3, len(kernel_data4)):
                group_data4.append(kernel_data4[i])    
            idx += 1

        elif idx % 5 == 0:
            m_name5 = row[0]
            gpu5 = row[1]
            batch5 = row[2]
            total_latency5 = row[3]
            exec_times5 = row[4]
            avg_latency5 = row[5]
            group_data5 = [m_name5, gpu5, batch5, total_latency5, exec_times5, avg_latency5]

            # 核函数相关数据获取
            key = m_name5 + "_b" + batch5 + "_g" + gpu5
            kernel_data5 = model_kernel_data[key]
            # 从第四个数据开始append，因为前三个是模型名、批量、GPU资源量
            for i in range(3, len(kernel_data5)):
                group_data5.append(kernel_data5[i])    
            
            idx = 1

            write_line = group_data1 + [""] + group_data2 + [""] + group_data3 + [""] +\
                         group_data4 + [""] + group_data5
            write_lines.append(write_line)

# 写结果部分
res_file = "./pro_" + target_file
with open(res_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for row in write_lines:
        writer.writerow(row)
