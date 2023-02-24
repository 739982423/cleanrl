import csv
import os

two_models_origin_data_file = ["pro_a_i.csv", "pro_a_m.csv","pro_a_r.csv","pro_a_v.csv","pro_d_a.csv",
                               "pro_d_i.csv","pro_d_m.csv","pro_d_r.csv","pro_d_v.csv","pro_m_i.csv",
                               "pro_r_i.csv","pro_r_m.csv","pro_v_i.csv","pro_v_m.csv","pro_v_r.csv"]
three_models_origin_data_file = ["res_tmp_d_r_v.csv"]
four_models_origin_data_file = ["res_tmp_d_m_r_v.csv"]

two_models_origin_data = []
three_models_origin_data = []
four_models_origin_data = []

# 读入两个模型共存时的原始数据(一行是一次共存执行的数据)
for file in two_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            two_models_origin_data.append(row)

# 读入三个模型共存时的原始数据(一行是一次共存执行的数据)
for file in three_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            three_models_origin_data.append(row)

# 读入四个模型共存时的原始数据(一行是一次共存执行的数据)
for file in four_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            four_models_origin_data.append(row)

# ------------------------------ 决策树所需输入 --------------------------------
two_models_input = []
two_models_input = []
two_models_input = []