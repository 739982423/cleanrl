import csv
import os
import collections

def clean_number(string):
    if string == "nan":
        return 0
    ret = ""
    for s in string:
        if s >= '0' and s <= '9':
            ret += s
    return int(ret)

def get_model_single_latency(model_list, path, store_dict):
    for model_name in model_list:
        file = path + "/" + model_name + ".csv"
        print(file)
        with open(file, mode="r", encoding="gbk") as f:
            reader = csv.reader(f)
            line = 0
            for row in reader:
                line += 1
                for idx, b in enumerate(["1","4","8","12","16"]):
                    key = model_name + "_b" + b + "_g" + str(line * 10)
                    store_dict[key] = float(row[idx])

dirs = os.listdir("./profile")

model_message_dict = dict()
get_model_single_latency(["alexnet","resnet50","mobilenet","inception","vgg19","densenet201"], "./cart/1", model_message_dict)

write_lines = [["Model", "Batch", "GPU Resource", "L1RL2", "L1WL2", "L2RD", "L2WD", "L2Hit", "L2Miss", "Latency"]]
for dir in dirs:
    if dir[-4:] != ".csv":
        continue
    with open("./profile/" + dir, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = 0
        m_name, batch, gpu_r = dir[:-4].split("_")
        batch = batch[1:]
        gpu_r = gpu_r[1:]
        print(m_name, batch, gpu_r)
        # 这里的latency是模型单独执行的时延，在cart/1文件夹里有每个模型的数据，batch=4,8,12,16，gpu资源量从10-100十组
        latency = model_message_dict[m_name + "_b" + batch + "_g" + gpu_r]
        
        # 初始化
        L1RL2 = 0
        L1WL2 = 0
        L2RD = 0
        L2WD = 0
        L2Hit = 0
        L2Miss = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            L1RL2 += clean_number(row[465])    # lts__t_sectors_srcunit_tex_op_read.sum [sector]
            L1WL2 += clean_number(row[479])    # lts__t_sectors_srcunit_tex_op_write.sum [sector]
            L2RD += clean_number(row[191])     # dram__sectors_read.sum [sector]
            L2WD += clean_number(row[192])     # dram__sectors.write.sum [sector]
            L2Hit += clean_number(row[409])    # lts__t_sectors_lookup_hit.sum [sector]
            L2Miss += clean_number(row[410])  # lts__t_sectors_lookup_miss.sum [sector]
        write_lines.append([m_name, batch, gpu_r, L1RL2, L1WL2, L2RD, L2WD, L2Hit, L2Miss, latency])

summary_file = "./summary3.csv"
with open(summary_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for row in write_lines:
        writer.writerow(row)


