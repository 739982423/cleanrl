import csv
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
import random
import joblib

two_models_origin_data_file = ["pro_a_i.csv", "pro_a_m.csv","pro_a_r.csv","pro_a_v.csv","pro_d_a.csv",
                               "pro_d_i.csv","pro_d_m.csv","pro_d_r.csv","pro_d_v.csv","pro_m_i.csv",
                               "pro_r_i.csv","pro_r_m.csv","pro_v_i.csv","pro_v_m.csv","pro_v_r.csv"]
three_models_origin_data_file = ["pro_cleaned_d_r_v.csv"]
four_models_origin_data_file = ["pro_cleaned_d_m_r_v.csv"]

two_models_origin_data = []
three_models_origin_data = []
four_models_origin_data = []

# 读入两个模型共存时的原始数据(一行是一次共存执行的数据)
for file in two_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            two_models_origin_data.append(row)

# 读入三个模型共存时的原始数据(一行是一次共存执行的数据)
for file in three_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            three_models_origin_data.append(row)

# 读入四个模型共存时的原始数据(一行是一次共存执行的数据)
for file in four_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line < 1:
                line += 1
                continue
            four_models_origin_data.append(row)

# ------------------------------ 决策树所需输入：11项 --------------------------------
two_models_input_x = []
two_models_input_y = []
three_models_input_x = []
three_models_input_y = []
four_models_input_x = []
four_models_input_y = []

for row in two_models_origin_data:
    m1_name = row[0]
    m1_g = float(row[1])
    m1_b = float(row[2])
    m1_increased_latency = float(row[5])
    m1_single_latency = float(row[12])
    m1_L1RL2 = float(row[6]) / m1_single_latency
    m1_L1WL2 = float(row[7]) / m1_single_latency
    m1_L2RD = float(row[8]) / m1_single_latency
    m1_L2WD  = float(row[9]) / m1_single_latency
    m1_L2Hit = float(row[10]) / m1_single_latency
    m1_L2Miss = float(row[11]) / m1_single_latency
    

    m2_name = row[14]
    m2_g = float(row[15])
    m2_b = float(row[16])
    m2_increased_latency = float(row[19])
    m2_single_latency = float(row[26])
    m2_L1RL2 = float(row[20]) / m2_single_latency
    m2_L1WL2 = float(row[21]) / m2_single_latency
    m2_L2RD = float(row[22]) / m2_single_latency
    m2_L2WD  = float(row[23]) / m2_single_latency
    m2_L2Hit = float(row[24]) / m2_single_latency
    m2_L2Miss = float(row[25]) / m2_single_latency
    
    m1_hit_rate = m1_L2Hit / (m1_L2Hit + m1_L2Miss)
    m2_hit_rate = m2_L2Hit / (m2_L2Hit + m2_L2Miss)
    m1_increased_latency_percent = (m1_increased_latency - m1_single_latency) / m1_single_latency
    m2_increased_latency_percent = (m2_increased_latency - m2_single_latency) / m2_single_latency

    # m1_input_line的意思是m1作为预测模型时的CART输入
    m1_input_line = [m1_L1RL2, m1_L1WL2, m1_L2RD, m1_L2WD, m1_L2Hit, m1_L2Miss, m1_hit_rate,
                     m2_L1RL2, m2_L1WL2, m2_L2RD, m2_L2WD, m2_L2Hit, m2_L2Miss]
    
    # m2_input_line的意思是m2作为预测模型时的CART输入
    m2_input_line = [m2_L1RL2, m2_L1WL2, m2_L2RD, m2_L2WD, m2_L2Hit, m2_L2Miss, m2_hit_rate,
                     m1_L1RL2, m1_L1WL2, m1_L2RD, m1_L2WD, m1_L2Hit, m1_L2Miss]
    
    # 添加m1作为预测模型时的输入X和标签Y
    two_models_input_x.append(m1_input_line)
    two_models_input_y.append(m1_increased_latency_percent)

    # 添加m2作为预测模型时的输入X和标签Y
    two_models_input_x.append(m2_input_line)
    two_models_input_y.append(m2_increased_latency_percent)

for row in three_models_origin_data:
    m1_name = row[0]
    m1_g = float(row[1])
    m1_b = float(row[2])
    m1_increased_latency = float(row[5])
    m1_L1RL2 = float(row[6]) / m1_single_latency
    m1_L1WL2 = float(row[7]) / m1_single_latency
    m1_L2RD = float(row[8]) / m1_single_latency
    m1_L2WD  = float(row[9]) / m1_single_latency
    m1_L2Hit = float(row[10]) / m1_single_latency
    m1_L2Miss = float(row[11]) / m1_single_latency
    m1_single_latency = float(row[12])

    m2_name = row[14]
    m2_g = float(row[15])
    m2_b = float(row[16])
    m2_increased_latency = float(row[19])
    m2_single_latency = float(row[26])
    m2_L1RL2 = float(row[20]) / m2_single_latency
    m2_L1WL2 = float(row[21]) / m2_single_latency
    m2_L2RD = float(row[22]) / m2_single_latency
    m2_L2WD  = float(row[23]) / m2_single_latency
    m2_L2Hit = float(row[24]) / m2_single_latency
    m2_L2Miss = float(row[25]) / m2_single_latency
    
    m3_name = row[28]
    m3_g = float(row[29])
    m3_b = float(row[30])
    m3_increased_latency = float(row[33])
    m3_single_latency = float(row[40])
    m3_L1RL2 = float(row[34]) / m3_single_latency
    m3_L1WL2 = float(row[35]) / m3_single_latency
    m3_L2RD = float(row[36]) / m3_single_latency
    m3_L2WD  = float(row[37]) / m3_single_latency
    m3_L2Hit = float(row[38]) / m3_single_latency
    m3_L2Miss = float(row[39]) / m3_single_latency
    

    m1_hit_rate = m1_L2Hit / (m1_L2Hit + m1_L2Miss)
    m2_hit_rate = m2_L2Hit / (m2_L2Hit + m2_L2Miss)
    m3_hit_rate = m3_L2Hit / (m3_L2Hit + m3_L2Miss)

    m1_increased_latency_percent = (m1_increased_latency - m1_single_latency) / m1_single_latency
    m2_increased_latency_percent = (m2_increased_latency - m2_single_latency) / m2_single_latency
    m3_increased_latency_percent = (m3_increased_latency - m3_single_latency) / m3_single_latency

    # 添加m1作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m2_L1RL2  + m3_L1RL2
    colocated_L1WL2  = m2_L1WL2  + m3_L1WL2
    colocated_L2RD   = m2_L2RD   + m3_L2RD 
    colocated_L2WD   = m2_L2WD   + m3_L2WD 
    colocated_L2Hit  = m2_L2Hit  + m3_L2Hit
    colocated_L2Miss = m2_L2Miss + m3_L2Miss
    m1_input_line = [m1_L1RL2, m1_L1WL2, m1_L2RD, m1_L2WD, m1_L2Hit, m1_L2Miss, m1_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    three_models_input_x.append(m1_input_line)
    three_models_input_y.append(m1_increased_latency_percent)

    # 添加m2作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m1_L1RL2  + m3_L1RL2
    colocated_L1WL2  = m1_L1WL2  + m3_L1WL2
    colocated_L2RD   = m1_L2RD   + m3_L2RD 
    colocated_L2WD   = m1_L2WD   + m3_L2WD 
    colocated_L2Hit  = m1_L2Hit  + m3_L2Hit
    colocated_L2Miss = m1_L2Miss + m3_L2Miss
    m2_input_line = [m2_L1RL2, m2_L1WL2, m2_L2RD, m2_L2WD, m2_L2Hit, m2_L2Miss, m2_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    three_models_input_x.append(m2_input_line)
    three_models_input_y.append(m2_increased_latency_percent)

    # 添加m3作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m1_L1RL2  + m2_L1RL2
    colocated_L1WL2  = m1_L1WL2  + m2_L1WL2
    colocated_L2RD   = m1_L2RD   + m2_L2RD 
    colocated_L2WD   = m1_L2WD   + m2_L2WD 
    colocated_L2Hit  = m1_L2Hit  + m2_L2Hit
    colocated_L2Miss = m1_L2Miss + m2_L2Miss
    m3_input_line = [m3_L1RL2, m3_L1WL2, m3_L2RD, m3_L2WD, m3_L2Hit, m3_L2Miss, m3_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    three_models_input_x.append(m3_input_line)
    three_models_input_y.append(m3_increased_latency_percent)

for row in four_models_origin_data:
    m1_name = row[0]
    m1_g = float(row[1])
    m1_b = float(row[2])
    m1_increased_latency = float(row[5])
    m1_single_latency = float(row[12])
    m1_L1RL2 = float(row[6]) / m1_single_latency
    m1_L1WL2 = float(row[7]) / m1_single_latency
    m1_L2RD = float(row[8]) / m1_single_latency
    m1_L2WD  = float(row[9]) / m1_single_latency
    m1_L2Hit = float(row[10]) / m1_single_latency
    m1_L2Miss = float(row[11]) / m1_single_latency
    
    m2_name = row[14]
    m2_g = float(row[15])
    m2_b = float(row[16])
    m2_increased_latency = float(row[19])
    m2_single_latency = float(row[26])
    m2_L1RL2 = float(row[20]) / m2_single_latency
    m2_L1WL2 = float(row[21]) / m2_single_latency
    m2_L2RD = float(row[22]) / m2_single_latency
    m2_L2WD  = float(row[23]) / m2_single_latency
    m2_L2Hit = float(row[24]) / m2_single_latency
    m2_L2Miss = float(row[25]) / m2_single_latency
    
    m3_name = row[28]
    m3_g = float(row[29])
    m3_b = float(row[30])
    m3_increased_latency = float(row[33])
    m3_single_latency = float(row[40])
    m3_L1RL2 = float(row[34]) / m3_single_latency
    m3_L1WL2 = float(row[35]) / m3_single_latency
    m3_L2RD = float(row[36]) / m3_single_latency
    m3_L2WD  = float(row[37]) / m3_single_latency
    m3_L2Hit = float(row[38]) / m3_single_latency
    m3_L2Miss = float(row[39]) / m3_single_latency
    
    m4_name = row[28]
    m4_g = float(row[29])
    m4_b = float(row[30])
    m4_increased_latency = float(row[33])
    m4_single_latency = float(row[40])
    m4_L1RL2 = float(row[34]) / m4_single_latency
    m4_L1WL2 = float(row[35]) / m4_single_latency
    m4_L2RD = float(row[36]) / m4_single_latency
    m4_L2WD  = float(row[37]) / m4_single_latency
    m4_L2Hit = float(row[38]) / m4_single_latency
    m4_L2Miss = float(row[39]) / m4_single_latency

    m1_hit_rate = m1_L2Hit / (m1_L2Hit + m1_L2Miss)
    m2_hit_rate = m2_L2Hit / (m2_L2Hit + m2_L2Miss)
    m3_hit_rate = m3_L2Hit / (m3_L2Hit + m3_L2Miss)
    m4_hit_rate = m4_L2Hit / (m4_L2Hit + m4_L2Miss)

    m1_increased_latency_percent = (m1_increased_latency - m1_single_latency) / m1_single_latency
    m2_increased_latency_percent = (m2_increased_latency - m2_single_latency) / m2_single_latency
    m3_increased_latency_percent = (m3_increased_latency - m3_single_latency) / m3_single_latency
    m4_increased_latency_percent = (m4_increased_latency - m4_single_latency) / m4_single_latency

    # 添加m1作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m2_L1RL2  + m3_L1RL2  + m4_L1RL2
    colocated_L1WL2  = m2_L1WL2  + m3_L1WL2  + m4_L1WL2 
    colocated_L2RD   = m2_L2RD   + m3_L2RD   + m4_L2RD
    colocated_L2WD   = m2_L2WD   + m3_L2WD   + m4_L2WD
    colocated_L2Hit  = m2_L2Hit  + m3_L2Hit  + m4_L2Hit
    colocated_L2Miss = m2_L2Miss + m3_L2Miss + m4_L2Miss
    m1_input_line = [m1_L1RL2, m1_L1WL2, m1_L2RD, m1_L2WD, m1_L2Hit, m1_L2Miss, m1_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    four_models_input_x.append(m1_input_line)
    four_models_input_y.append(m1_increased_latency_percent)

    # 添加m2作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m1_L1RL2  + m3_L1RL2  + m4_L1RL2
    colocated_L1WL2  = m1_L1WL2  + m3_L1WL2  + m4_L1WL2
    colocated_L2RD   = m1_L2RD   + m3_L2RD   + m4_L2RD
    colocated_L2WD   = m1_L2WD   + m3_L2WD   + m4_L2WD
    colocated_L2Hit  = m1_L2Hit  + m3_L2Hit  + m4_L2Hit
    colocated_L2Miss = m1_L2Miss + m3_L2Miss + m4_L2Miss
    m2_input_line = [m2_L1RL2, m2_L1WL2, m2_L2RD, m2_L2WD, m2_L2Hit, m2_L2Miss, m2_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    four_models_input_x.append(m2_input_line)
    four_models_input_y.append(m2_increased_latency_percent)

    # 添加m3作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m1_L1RL2  + m2_L1RL2  + m4_L1RL2
    colocated_L1WL2  = m1_L1WL2  + m2_L1WL2  + m4_L1WL2
    colocated_L2RD   = m1_L2RD   + m2_L2RD   + m4_L2RD
    colocated_L2WD   = m1_L2WD   + m2_L2WD   + m4_L2WD
    colocated_L2Hit  = m1_L2Hit  + m2_L2Hit  + m4_L2Hit
    colocated_L2Miss = m1_L2Miss + m2_L2Miss + m4_L2Miss
    m3_input_line = [m3_L1RL2, m3_L1WL2, m3_L2RD, m3_L2WD, m3_L2Hit, m3_L2Miss, m3_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    four_models_input_x.append(m3_input_line)
    four_models_input_y.append(m3_increased_latency_percent)

    # 添加m4作为预测模型时的输入X和标签Y
    colocated_L1RL2  = m2_L1RL2  + m3_L1RL2  + m1_L1RL2
    colocated_L1WL2  = m2_L1WL2  + m3_L1WL2  + m1_L1WL2
    colocated_L2RD   = m2_L2RD   + m3_L2RD   + m1_L2RD
    colocated_L2WD   = m2_L2WD   + m3_L2WD   + m1_L2WD
    colocated_L2Hit  = m2_L2Hit  + m3_L2Hit  + m1_L2Hit
    colocated_L2Miss = m2_L2Miss + m3_L2Miss + m1_L2Miss
    m4_input_line = [m4_L1RL2, m4_L1WL2, m4_L2RD, m4_L2WD, m4_L2Hit, m4_L2Miss, m4_hit_rate,
                     colocated_L1RL2, colocated_L1WL2, colocated_L2RD, colocated_L2WD, colocated_L2Hit, colocated_L2Miss]
    four_models_input_x.append(m4_input_line)
    four_models_input_y.append(m4_increased_latency_percent)

# CART生成部分

filtered_data_x = []
filtered_data_y = []

# 加入两模型共存数据
for idx, data in enumerate(two_models_input_x):
    filtered_data_x.append(data)
    filtered_data_y.append(two_models_input_y[idx])

# 加入三模型共存数据
for idx, data in enumerate(three_models_input_x):
    filtered_data_x.append(data)
    filtered_data_y.append(three_models_input_y[idx])

# 加入四模型共存数据
for idx, data in enumerate(four_models_input_x):
    filtered_data_x.append(data)
    filtered_data_y.append(four_models_input_y[idx])

res_state = 1
x_train, x_test, y_train, y_test = train_test_split(filtered_data_x, filtered_data_y, test_size=0.2, random_state=res_state)

dtr = tree.DecisionTreeRegressor(max_depth=15, min_samples_split=5)
dtr.fit(x_train, y_train)
dtr_predict = dtr.predict(x_test)
print("训练完成后初次测试的结果：")
print("curstate =", res_state)
print('R-squared value of DecisionTreeRegressor:', dtr.score(x_test, y_test))
print('The mean squared error of DecisionTreeRegressor:',mean_squared_error(y_test,dtr_predict))
print('The mean absolute error of DecisionTreeRegressor:',mean_absolute_error(y_test,dtr_predict))
print("---------------------------------------------------------------")

# high_R2 = []
# min_R2 = 0
# for i in range(1, 999):
#     print(i, "...")
#     x_train, x_test, y_train, y_test = train_test_split(filtered_data_x, filtered_data_y, test_size=0.2, random_state=i)
#     dtr = tree.DecisionTreeRegressor(max_depth=15, min_samples_split=5)
#     dtr.fit(x_train, y_train)
#     cur_R2 = dtr.score(x_test, y_test)
#     if cur_R2 > min_R2:
#         res_state = i
#         min_R2 = cur_R2
#         print("new state:", res_state)
#         print("cur_R2:", cur_R2)
#         high_R2.append((i, cur_R2))

# -------------- 按大小顺序排序后画图的代码 ----------------
feat_importance = dtr.tree_.compute_feature_importances(normalize=True)
print("Gini Importance:", feat_importance)

plt.figure()
plt.title("Kernel Feature importance")
plt.xlabel("Gini Importance")
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

names = ['L1RL2','L1WL2','L2RD','L2WD',
         'L2 Hit', 'L2 Miss', 'L2 Hit Rate',
         'L1RL2_','L1WL2_','L2RD_','L2WD_',
         'L2 Hit_', 'L2 Miss_',]
pair = []
for i in range(len(feat_importance)):
    pair.append((feat_importance[i], names[i]))

pair.sort(key = lambda x:x[0])

predict_model_data = [[],[]]
colocated_model_data = [[],[]]
plt.grid(linestyle=":", axis="y")


predict_model_legend_figured = False
colocated_model_legend_figured = False

for i in range(len(pair)):
    val, name = pair[i][0], pair[i][1]
    if i == len(pair) - 1:
        val *= 0.6
    if name[-1] != "_":
        if predict_model_legend_figured == False:
            plt.barh(name, val, 0.7, edgecolor="black", color="dodgerblue",
                    label="Predict Models", lw=1)
            predict_model_legend_figured = True
        else:
            plt.barh(name, val, 0.7, edgecolor="black", color="dodgerblue", lw=1)
    else:
        if colocated_model_legend_figured == False:
            plt.barh(name, val, 0.7, edgecolor="black", color="orange",
                    label="Other Co-located Models", lw=1)
            colocated_model_legend_figured = True
        else:
            plt.barh(name, val, 0.7, edgecolor="black", color="orange", lw=1)
plt.xticks(fontsize = 8)
plt.legend()
plt.tight_layout()
plt.show()

# -------------- 保存CART模型 ---------------
joblib_file = "cart.pkl"
joblib.dump(dtr, joblib_file)     # 存储
new_dtr = joblib.load(joblib_file) # 读取
new_dtr_y_predict = new_dtr.predict(x_test, y_test)
print("保存CART后重新读取，再次测试的结果：")
print('R-squared value of DecisionTreeRegressor:', new_dtr.score(x_test, y_test))
print('The mean squared error of DecisionTreeRegressor:', mean_squared_error(y_test,new_dtr_y_predict))
print('The mean absolute error of DecisionTreeRegressor:', mean_absolute_error(y_test,new_dtr_y_predict))