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

dirs = os.listdir()
for dir in dirs:
    if dir[-4:] != ".csv":
        continue
    print(dir)
    write_lines = [["model1 name","gpu1","batch1","total latency","exec times", "avg latency",
                   "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency", "",
                   "model2 name","gpu2","batch2","total latency","exec times", "avg latency",
                   "L1RL2","L1WL2","L2RD","L2WD","L2Hit","L2Miss","single latency"]]
    with open(dir, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        group_data1 = []
        group_data2 = []
        group_cnt = 0
        idx = 1
        for row in reader:
            if idx % 4 == 1:
                idx += 1
                continue
            elif idx % 3 == 2:
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

            elif idx % 4 == 3:
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
                
                write_line = group_data1 + [""] + group_data2
                write_lines.append(write_line)

                idx += 1

            elif idx % 4 == 0:
                group_data1 = []
                group_data2 = []
                group_cnt += 1
                
                idx = 1
                if row != []:
                    print("group error, idx = {}, group_cnt = {}".format(idx, group_cnt))

    # 写结果部分
    res_file = "./filtered/pro_" + dir
    with open(res_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in write_lines:
            writer.writerow(row)
