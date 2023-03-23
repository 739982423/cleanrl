# 该文件测试pandas各项功能
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import csv

df = pd.read_csv("../minworkload/res50.csv")
print(df.head())    # 查看数据前几行
print(df.shape)     # 查看数据的形状，返回行数，列数
print(df.columns)   # 查看列名列表
print(df.index)     # 查看索引列
print(df.dtypes)    # 查看每列数据类型