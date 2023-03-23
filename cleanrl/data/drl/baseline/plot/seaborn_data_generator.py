# 该文件为绘制第三章结果图准备seaborn绘图格式的数据集，主要工作是将baseline的横线对应的reward值，在各个横坐标x点上都记录保存为csv
# 使得seaborn绘制的baseline横线也带有误差带
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import csv

df = pd.read_csv("../minworkload/res3.csv")
print(df.head())    # 查看数据前几行
print(df['0'][0])
for col in df.columns:
    series = df[col]
    new_df = pd.DataFrame({'data':[0 for _ in range(1800)], 'x':[0 for _ in range(1800)]})
    line_cnt = 0
    for loop_cnt in range(600):
        for data in series:
            new_df['data'][line_cnt] = data
            new_df['x'][line_cnt] = loop_cnt
            line_cnt += 1
    new_df.to_csv('res_ascend{}.csv'.format(col))
    # break