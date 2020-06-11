import itertools
from datetime import datetime as dt
import numpy as np
import pandas as pd

import json
# 读取原始数据
loc = "./football/"

raw_data_1 = pd.read_csv(loc + '2005-06.csv')

# raw_data_1.head()
# raw_data_1.tail()
# 挑选信息列
# HomeTeam 主场球队名
# AwayTeam 客场球队名

# FTHG 是 全场 主场球队进球数
# FTAG 是 全场 客场球队进球数

# FTR 是 比赛结果 (H=主场赢, D=平局, A=客场赢)
columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

playing_statistics_1 = raw_data_1[columns_req]

# print(playing_statistics_1)

# 计算每个队周累计净胜球数量


def get_test_diff(playing_stat):
    # 创建一个字典，每个 team 的 name 作为 key
    results = {}  # 比赛情况列表
    points = {}  # 得分情况
    ags = {}  # 总进球
    algs = {}  # 总失球
    gds = {}  # 净胜球列表
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        results[i] = []
        points[i] = []
        ags[i] = []
        algs[i] = []
        gds[i] = []
    # 对于每一场比赛
    for i in range(len(playing_stat)):
        # 全场比赛，主场队伍的进球数
        HTGS = playing_stat.iloc[i]['FTHG']
        # 全场比赛，客场队伍的进球数
        ATGS = playing_stat.iloc[i]['FTAG']

        ags[playing_stat.iloc[i].HomeTeam].append(HTGS)
        algs[playing_stat.iloc[i].HomeTeam].append(ATGS)
        ags[playing_stat.iloc[i].AwayTeam].append(ATGS)
        algs[playing_stat.iloc[i].AwayTeam].append(HTGS)

        # 把主场队伍的净胜球数添加到 team 这个 字典中对应的主场队伍下
        gds[playing_stat.iloc[i].HomeTeam].append(HTGS-ATGS)
        # 把客场队伍的净胜球数添加到 team 这个 字典中对应的客场队伍下
        gds[playing_stat.iloc[i].AwayTeam].append(ATGS-HTGS)

        if playing_stat.iloc[i].FTR == 'H':
            # 主场 赢，则主场记为赢，客场记为输
            results[playing_stat.iloc[i].HomeTeam].append('W')
            points[playing_stat.iloc[i].HomeTeam].append(3)
            results[playing_stat.iloc[i].AwayTeam].append('L')
            points[playing_stat.iloc[i].AwayTeam].append(0)
        elif playing_stat.iloc[i].FTR == 'A':
            # 客场 赢，则主场记为输，客场记为赢
            results[playing_stat.iloc[i].AwayTeam].append('W')
            points[playing_stat.iloc[i].AwayTeam].append(3)
            results[playing_stat.iloc[i].HomeTeam].append('L')
            points[playing_stat.iloc[i].HomeTeam].append(0)
        else:
            # 平局
            results[playing_stat.iloc[i].AwayTeam].append('D')
            points[playing_stat.iloc[i].AwayTeam].append(1)
            results[playing_stat.iloc[i].HomeTeam].append('D')
            points[playing_stat.iloc[i].HomeTeam].append(1)
    # print(results)
    # print(points)
    # print(results[playing_stat.iloc[i].HomeTeam])
    # print(ags)
    # print(algs)

    name = []
    winNum = {}
    drawNum = {}
    loseNum = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        name.append(i)
        winNum[i] = []
        drawNum[i] = []
        loseNum[i] = []
    # print(name)

    all = []
    data = []

    for i in range(0, 38):
        for j in range(0, len(name)):
            if results[name[j]][i] == 'W':
                winNum[name[j]].append(1)
            else:
                winNum[name[j]].append(0)
            if results[name[j]][i] == 'D':
                drawNum[name[j]].append(1)
            else:
                drawNum[name[j]].append(0)
            if results[name[j]][i] == 'L':
                loseNum[name[j]].append(1)
            else:
                loseNum[name[j]].append(0)

            if i == 0:
                ags[name[j]][i] = ags[name[j]][i]
                algs[name[j]][i] = algs[name[j]][i]
                gds[name[j]][i] = gds[name[j]][i]
                points[name[j]][i] = points[name[j]][i]
                winNum[name[j]][i] = winNum[name[j]][i]
                drawNum[name[j]][i] = drawNum[name[j]][i]
                loseNum[name[j]][i] = loseNum[name[j]][i]
            else:
                ags[name[j]][i] = ags[name[j]][i] + ags[name[j]][i-1]
                algs[name[j]][i] = algs[name[j]][i] + algs[name[j]][i-1]
                gds[name[j]][i] = gds[name[j]][i] + gds[name[j]][i-1]
                points[name[j]][i] = points[name[j]][i] + points[name[j]][i-1]
                winNum[name[j]][i] = winNum[name[j]][i] + winNum[name[j]][i-1]
                drawNum[name[j]][i] = drawNum[name[j]][i] + drawNum[name[j]][i-1]
                loseNum[name[j]][i] = loseNum[name[j]][i] + loseNum[name[j]][i-1]
            firstMatchResult = ''
            secondeMatchResult = ''
            threeMatchResult = ''

            if i > 2:
                threeMatchResult = results[name[j]][i-3]
                secondeMatchResult = results[name[j]][i-2]
                firstMatchResult = results[name[j]][i-1]

            if i < 37:
                nextResult = results[name[j]][i+1]
            else:
                nextResult = ''

            let = {
                'id': str(i+1),
                'name': name[j],
                'result': results[name[j]][i],  # 当前场次比赛结果
                'nextResult': nextResult,
                'ags': int(ags[name[j]][i]),  # 截止目前总进球
                'algs': int(algs[name[j]][i]),  # 截止目前总失球
                'gds': int(gds[name[j]][i]),  # 截止目前总净胜球
                # 截止当前平均净胜球
                'averGd': float('%.2f' % (gds[name[j]][i] / (i+1))),
                'points': int(points[name[j]][i]),  # 截止目前总分数
                # 截止当前平均得分
                'averPoint': float('%.2f' % (points[name[j]][i] / (i+1))),
                'lastOneMatch': firstMatchResult,
                'lastTwoMatch': secondeMatchResult,
                'lastThreeMatch': threeMatchResult,
                'ottMatch': firstMatchResult + secondeMatchResult + threeMatchResult,
                'winNum': int(winNum[name[j]][i]),  # 截止目前胜场
                'drawNum': int(drawNum[name[j]][i]),  # 截止目前平场
                'loseNum': int(loseNum[name[j]][i])  # 截止目前负场
                # 'aver': float('%.2f' % (teams[name[j]][i] / (i+1)))  # 两位小数
            }
            data.append(let)
        all.append({
            'week': str(i+1),
            'weekData': data
        })
        data = []
    # print(points)
    # print(all)  # all为20支队伍截止到每轮的总净胜球、平均净胜球
    with open('./files/all.json', 'w') as f:
        json.dump(all, f)

    # 要做每轮累积总得分，平均每周得分、截止到每轮的总进球，失球、胜负平场次


if __name__ == '__main__':
    get_test_diff(playing_statistics_1)
