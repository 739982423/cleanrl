# 本文件用来删除d_m_r_v.csv和d_r_v.csv两个文件中的无效数据的（无效数据指的是4模型的测试最后只返回了少于4个模型的共存数据，这样的一次执行数据，在文件中表现为结果行数少于4，应删除）
# 本文件将处理后的结果保存为cleaned_d_m_r_v.csv和cleaned_d_r_v.csv
import csv
import os

# target_file = "d_m_r_v.csv"
target_file = "d_r_v.csv"

group_length = 6 if target_file == "d_m_r_v.csv" else 5

write_lines = []
with open(target_file, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    group_rows = []
    cnt = 0
    idx = 0
    for row in reader:
        # print(idx)
        # print(row)
        idx += 1
        if len(row) >= 1 and row[0] == 'model name':
            group_rows.append(row)
            if cnt < group_length:
                group_rows = []
                cnt = 0
            else:
                for row2 in group_rows:
                    write_lines.append(row2)
                group_rows = []
                cnt = 0
        else:
            if target_file == "d_m_r_v.csv":
                if row != ['','','','','','']:
                    group_rows.append(row)
                cnt += 1
            else:
                if row != ['','',''] and row != []:
                    group_rows.append(row)
                cnt += 1 

with open("cleaned_" + target_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in write_lines:
            writer.writerow(row)