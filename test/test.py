import numpy as np
import pandas as pd

import json

import csv

# 读取原始数据
loc = "./football/"

csv_reader = csv.reader(open(loc + '2019-20.csv', encoding='utf-8'))
n = len(list(csv_reader))
dataNum = int((n-1) / 10)
print(dataNum)

# 思路：先获取目前的轮数，比如28轮。再遍历找每个让每个球队出现28次，出现29次把那行数据删掉

raw_data_1 = pd.read_csv(loc + '2019-20.csv')
columns_req = ['HomeTeam', 'AwayTeam']

playing_statistics_1 = raw_data_1[columns_req]
print(playing_statistics_1)
Teams = []

for i in range(len(playing_statistics_1)):
    Teams.append(playing_statistics_1.iloc[i]['HomeTeam'])
    # 把所有球队遍历在Teams里，求索引
    Teams.append(playing_statistics_1.iloc[i]['AwayTeam'])
# print(Teams) # 576，奇数是主队。偶数是客队；因为去掉了标题行，索引头为0
# print(len(Teams))


# 获取统计次数与所在索引
index = []  # 需要删除的索引

list1 = Teams
L1 = len(list1)
list2 = list(set(list1))
L2 = len(list2)  # 列表list2的长度
for m in range(L2):
    X = set()
    start = list1.index(list2[m])
    for n in range(L1):
        stop = L1
        if list2[m] in tuple(list1)[start:stop]:
            a = list1.index(list2[m], start, stop)
            X.add(a)
            start = start+1
    # print('元素：'+str(list2[m])+'，一共有'+str(len(X))+'个，在列表位置集合为：'+str(X))
    pos = []
    if len(X) > dataNum:
        for i in X:
            pos.append(i)
            pos.sort(reverse=False)
        # print(pos)
        # print(pos[len(pos)-1]) # 获取到最后一个index
        index.append(pos[len(pos)-1])
index.sort(reverse=False)

# 整理索引，客队减1与主队保持一致
for i in range(len(index)):
    if index[i] % 2 != 0:
        index[i] = index[i]-1
index = list(set(index))
print(index)

# 所有队伍成功
