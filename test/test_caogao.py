# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder in your jobs
import pandas as pd
# 读取原始数据
loc = "./football/"

raw_data_1 = pd.read_csv(loc + '2005-06.csv')
raw_data_2 = pd.read_csv(loc + '2006-07.csv')
raw_data_3 = pd.read_csv(loc + '2007-08.csv')
raw_data_4 = pd.read_csv(loc + '2008-09.csv')
raw_data_5 = pd.read_csv(loc + '2009-10.csv')
raw_data_6 = pd.read_csv(loc + '2010-11.csv')
raw_data_7 = pd.read_csv(loc + '2011-12.csv')
raw_data_8 = pd.read_csv(loc + '2012-13.csv')
raw_data_9 = pd.read_csv(loc + '2013-14.csv')
raw_data_10 = pd.read_csv(loc + '2014-15.csv')
raw_data_11 = pd.read_csv(loc + '2015-16.csv')
raw_data_12 = pd.read_csv(loc + '2016-17.csv')
raw_data_13 = pd.read_csv(loc + '2017-18.csv')
# 导入必须的包

import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
raw_data_1.head() #返回data的前几行数据，默认为前五行，需要前十行则data.head(10)
raw_data_1.tail() #返回data的后几行数据，默认为后五行，需要后十行则data.tail(10)
# 挑选信息列
# HomeTeam 主场球队名
# AwayTeam 客场球队名

# FTHG 是 全场 主场球队进球数
# FTAG 是 全场 客场球队进球数

# FTR 是 比赛结果 (H=主场赢, D=平局, A=客场赢)
columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

playing_statistics_1 = raw_data_1[columns_req]  # 取前五列数据，作为原始的特征数据
playing_statistics_2 = raw_data_2[columns_req]
playing_statistics_3 = raw_data_3[columns_req]
playing_statistics_4 = raw_data_4[columns_req]
playing_statistics_5 = raw_data_5[columns_req]
playing_statistics_6 = raw_data_6[columns_req]
playing_statistics_7 = raw_data_7[columns_req]
playing_statistics_8 = raw_data_8[columns_req]
playing_statistics_9 = raw_data_9[columns_req]
playing_statistics_10 = raw_data_10[columns_req]
playing_statistics_11 = raw_data_11[columns_req]
playing_statistics_12 = raw_data_12[columns_req]
playing_statistics_13 = raw_data_13[columns_req]
playing_statistics_1.head()


# 计算每个队周累计净胜球数量
def get_goals_diff(playing_stat):
    # print(playing_stat)
    # 创建一个字典，每个 team 的 name 作为 key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    # 对于每一场比赛
    # print(teams)
    for i in range(len(playing_stat)):
        # 全场比赛，主场队伍的进球数
        HTGS = playing_stat.iloc[i]['FTHG']  # 有多少行 i循环
        # 全场比赛，客场队伍的进球数
        ATGS = playing_stat.iloc[i]['FTAG']

        # 把主场队伍的净胜球数添加到 team 这个 字典中对应的主场队伍下
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS - ATGS)
        # 把客场队伍的净胜球数添加到 team 这个 字典中对应的客场队伍下
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS - HTGS)

    # 创建一个 GoalsScored 的 dataframe
    # 行是 team 列是 matchweek
    GoalsDifference = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    # GoalsDifference[0] = 0
    # 累加每个队的周比赛的净胜球数
    for i in range(2, 39):
        GoalsDifference[i] = GoalsDifference[i] + GoalsDifference[i - 1]
    
    GoalsDifference.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return GoalsDifference
    # print(GoalsDifference)

def get_gss(playing_stat):

    # 得到净胜球数统计
    GD = get_goals_diff(playing_stat)

    j = 0

    HTGD = [] # 主队（到目前这一场的净胜球，有n轮截止到n轮）
    ATGD = []

    # 全年一共380场比赛
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])

        if ((i + 1)% 10) == 0: # 一轮10场比赛，一轮完成之后取后面轮次的净胜球
            j = j + 1
        
    # 把每个队的 HTGS ATGS 信息补充到 dataframe 中
    playing_stat.loc[:,'HTGD'] = HTGD
    playing_stat.loc[:,'ATGD'] = ATGD

    return playing_stat
    # print(playing_stat)

playing_statistics_1 = get_gss(playing_statistics_1)
playing_statistics_2 = get_gss(playing_statistics_2)
playing_statistics_3 = get_gss(playing_statistics_3)
playing_statistics_4 = get_gss(playing_statistics_4)
playing_statistics_5 = get_gss(playing_statistics_5)
playing_statistics_6 = get_gss(playing_statistics_6)
playing_statistics_7 = get_gss(playing_statistics_7)
playing_statistics_8 = get_gss(playing_statistics_8)
playing_statistics_9 = get_gss(playing_statistics_9)
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)

playing_statistics_1.tail()
# 把比赛结果转换为得分，赢得三分，平局得一分，输不得分
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_matchres(playing_stat):
    # 创建一个字典，每个 team 的 name 作为 key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # 把比赛结果分别记录在主场队伍和客场队伍中
    # H：代表 主场 赢 A：代表 客场 赢 D：代表 平局
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            # 主场 赢，则主场记为赢，客场记为输
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            # 客场 赢，则主场记为输，客场记为赢
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            # 平局
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
        
    return pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    # print(pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T)

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2,39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    
    matchres_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return matchres_points
    # print(matchres_points)

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
    # 主场累计得分        
    playing_stat.loc[:,'HTP'] = HTP
    # 客场累计得分
    playing_stat.loc[:,'ATP'] = ATP
    return playing_stat
    # print(playing_stat)

playing_statistics_1 = get_agg_points(playing_statistics_1)
playing_statistics_2 = get_agg_points(playing_statistics_2)
playing_statistics_3 = get_agg_points(playing_statistics_3)
playing_statistics_4 = get_agg_points(playing_statistics_4)
playing_statistics_5 = get_agg_points(playing_statistics_5)
playing_statistics_6 = get_agg_points(playing_statistics_6)
playing_statistics_7 = get_agg_points(playing_statistics_7)
playing_statistics_8 = get_agg_points(playing_statistics_8)
playing_statistics_9 = get_agg_points(playing_statistics_9)
playing_statistics_10 = get_agg_points(playing_statistics_10)
playing_statistics_11 = get_agg_points(playing_statistics_11)
playing_statistics_12 = get_agg_points(playing_statistics_12)
playing_statistics_13 = get_agg_points(playing_statistics_13)
playing_statistics_1.tail()

# 上 n 次的比赛结果

# 例如：
# HM1 代表上次主场球队的比赛结果
# HM2 代表上上次主场球队的比赛结果

# AM1 代表上次客场球队的比赛结果
# AM2 代表上上次客场球队的比赛结果

def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final


def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    # print(form)
    # M 代表 unknown， 因为没有那么多历史
    h = ['M' for i in range(num * 10)]  
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num*10),380): # 上3次代表过了3轮
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
    
        past = form.loc[ht][j]               
        h.append(past[num-1])                    
    
        past = form.loc[at][j]            
        a.append(past[num-1]) 
    
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    return playing_statistics
    # print(playing_statistics)

playing_statistics_1 = add_form_df(playing_statistics_1)
playing_statistics_2 = add_form_df(playing_statistics_2)
playing_statistics_3 = add_form_df(playing_statistics_3)
playing_statistics_4 = add_form_df(playing_statistics_4)
playing_statistics_5 = add_form_df(playing_statistics_5)
playing_statistics_6 = add_form_df(playing_statistics_6)
playing_statistics_7 = add_form_df(playing_statistics_7)
playing_statistics_8 = add_form_df(playing_statistics_8)
playing_statistics_9 = add_form_df(playing_statistics_9)
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_1.tail()

# 加入比赛周特征（第几个比赛周）
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

playing_statistics_1 = get_mw(playing_statistics_1)
playing_statistics_2 = get_mw(playing_statistics_2)
playing_statistics_3 = get_mw(playing_statistics_3)
playing_statistics_4 = get_mw(playing_statistics_4)
playing_statistics_5 = get_mw(playing_statistics_5)
playing_statistics_6 = get_mw(playing_statistics_6)
playing_statistics_7 = get_mw(playing_statistics_7)
playing_statistics_8 = get_mw(playing_statistics_8)
playing_statistics_9 = get_mw(playing_statistics_9)
playing_statistics_10 = get_mw(playing_statistics_10)
playing_statistics_11 = get_mw(playing_statistics_11)
playing_statistics_12 = get_mw(playing_statistics_12)
playing_statistics_13 = get_mw(playing_statistics_13)
playing_statistics_1.tail()
# 拼接13个赛季的数据
playing_stat = pd.concat([playing_statistics_1,
                            playing_statistics_2,
                            playing_statistics_3,
                            playing_statistics_4,
                            playing_statistics_5,
                            playing_statistics_6,
                            playing_statistics_7,
                            playing_statistics_8,
                            playing_statistics_9,
                            playing_statistics_10,
                            playing_statistics_11,
                            playing_statistics_12,
                            playing_statistics_13], ignore_index=True)
# print(playing_stat)
# HTGD, ATGD ,HTP, ATP的值 除以 week 数，得到平均分
cols = ['HTGD','ATGD','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float) # 转化属性

# 把截止到目前比赛主客净胜球、累计得分转化为了，得到平均数
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW
playing_stat.tail()
playing_stat.keys()
#

# 抛弃前三周的比赛，抛弃队名和比赛周信息
# playing_stat = playing_stat[playing_stat.MW > 3]
playing_stat.drop(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'MW'],1, inplace=True)
playing_stat.keys()

# 比赛总数
n_matches = playing_stat.shape[0]

# 特征数
n_features = playing_stat.shape[1] - 1

# 主场获胜的数目
n_homewins = len(playing_stat[playing_stat.FTR == 'H'])

# 主场获胜的比例
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("比赛总数: {}".format(n_matches))
print("总特征数: {}".format(n_features))
print("主场胜利数: {}".format(n_homewins))
print("主场胜率: {:.2f}%".format(win_rate))

# 定义 target ，也就是否 主场赢
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'

playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)
playing_stat.to_csv(loc + "final_dataset.csv")
# 读取数据
playing_stat = pd.read_csv('./football/final_dataset.csv')
playing_stat.head()
# 删除读取
playing_stat.drop(['Unnamed: 0'],1, inplace=True)

playing_stat.head()
playing_stat.head()
# 数据分为特征和标签
X_all = playing_stat.drop(['FTR'],1)
y_all = playing_stat['FTR']

print(X_all)
print(y_all)

# if __name__ == '__main__':
    # get_goals_diff(playing_statistics_1)
    # get_gss(playing_statistics_1)
    # matchres = get_matchres(playing_statistics_1)
    # cum_pts = get_cuml_points(matchres)
    # print(matchres)
    # print(cum_pts)
    # get_agg_points(playing_statistics_1)
    # add_form_df(playing_statistics_1)
    # print(playing_statistics_1)