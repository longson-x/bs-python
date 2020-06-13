import itertools
from datetime import datetime as dt
import numpy as np
import pandas as pd

import json
import os
import csv
from pymongo import MongoClient as Client


def get_cnName(val, dataSet):
    for i in dataSet:
        if val == i['name']:
            return i['cnName']

# 读取原始数据

# raw_data_1.head()
# raw_data_1.tail()
# 挑选信息列
# HomeTeam 主场球队名
# AwayTeam 客场球队名

# FTHG 是 全场 主场球队进球数
# FTAG 是 全场 客场球队进球数

# FTR 是 比赛结果 (H=主场赢, D=平局, A=客场赢)


def get_season(country, season):
    loc = "./csv/%s/" % country

    csv_reader = csv.reader(open(loc + '%s.csv' % season, encoding='utf-8'))
    n = len(list(csv_reader))
    dataNum = int((n-1) / 10)

    raw_data = pd.read_csv(loc + '%s.csv' % season)  # 2005-06

    columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

    playing_statistics = raw_data[columns_req]

    get_data_diff(playing_statistics, country, season, dataNum)

# print(playing_statistics_1)

# 计算每个队周累计净胜球数量


def get_data_diff(playing_stat, country, season, dataNum):
    # 得到en-chinese的json
    en_chinese = []
    with open('./en-chinese/%s/%s_cname.json' %(country,country), 'r', encoding='utf-8') as fp:
        en_chinese = json.load(fp)

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

    all = [] #json文件调用
    all_db = [] #数据库调用
    data = []

    for i in range(0, dataNum):
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

            if i < dataNum-1:
                nextResult = results[name[j]][i+1]
            else:
                nextResult = ''

            let = {
                'weekId': str(i+1),
                'name': name[j],
                'cnName': get_cnName(name[j], en_chinese),
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

        data = sorted(data, key=lambda x: (-x['points'], -x['gds'], -x['ags'], -x['winNum'], -x['drawNum']))  # 排序

        all.append({
            'week': str(i+1),
            'weekData': data
        })
        all_db.append({
            'leagueName': returnLeagueName(country),
            'season': season,
            'week': str(i+1),
            'weekData': data
        })
        data = []
    # print(points)
    # print(all)  # all为20支队伍截止到每轮的总净胜球、平均净胜球
    myclient = Client('mongodb://localhost:27017/')
    mydb = myclient['bs_db']
    mycol = mydb[country]
    for item in all_db:
        mycol.insert_one(item)
    myclient.close()

    if not os.path.exists('./files/%s' % country):
        os.makedirs('./files/%s' % country)
    with open('./files/%s/%s.json' % (country, season), 'w', encoding='utf-8') as f:
        json.dump(all, f, ensure_ascii=False)

    # 要做每轮累积总得分，平均每周得分、截止到每轮的总进球，失球、胜负平场次

def returnLeagueName(eng):
    if eng == 'england':
        return '英格兰足球超级联赛'
    elif eng == 'italy':
        return '意大利足球甲级联赛'
    elif eng == 'spain':
        return '西班牙足球甲级联赛'
    elif eng == 'germany':
        return '德国足球甲级联赛'
    elif eng == 'france':
        return '法国足球甲级联赛'
    else: return ''

if __name__ == '__main__':
    year = ['2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20']
    for i in year:
        # get_season('england', i)
        # get_season('italy', i)
        # get_season('spain', i)
        # get_season('germany', i)
        get_season('france', i)

    # get_season('italy', '2016-17')
