# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder in your jobs
import pandas as pd
# 读取原始数据
loc = "./csv/england/"

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
raw_data_1.head()
raw_data_1.tail()
# 挑选信息列
# HomeTeam 主场球队名
# AwayTeam 客场球队名

# FTHG 是 全场 主场球队进球数
# FTAG 是 全场 客场球队进球数

# FTR 是 比赛结果 (H=主场赢, D=平局, A=客场赢)
columns_req = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR']

playing_statistics_1 = raw_data_1[columns_req]                      
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
    # 创建一个字典，每个 team 的 name 作为 key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []    
    # 对于每一场比赛
    for i in range(len(playing_stat)):
        # 全场比赛，主场队伍的进球数
        HTGS = playing_stat.iloc[i]['FTHG'] # 有多少行 i循环
        # 全场比赛，客场队伍的进球数
        ATGS = playing_stat.iloc[i]['FTAG']

        # 把主场队伍的净胜球数添加到 team 这个 字典中对应的主场队伍下
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS-ATGS)
        # 把客场队伍的净胜球数添加到 team 这个 字典中对应的客场队伍下
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS-HTGS)
    
    # 创建一个 GoalsScored 的 dataframe 
    # 行是 team 列是 matchweek
    GoalsDifference = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsDifference[0] = 0
    # 累加每个队的周比赛的净胜球数
    for i in range(2,39):
        GoalsDifference[i] = GoalsDifference[i] + GoalsDifference[i-1]
    
    return GoalsDifference # 返回20支球队38轮比赛之后的总净胜球（有n轮比赛就返回n）


def get_gss(playing_stat):

    # 得到净胜球数统计
    GD = get_goals_diff(playing_stat)

    j = 0

    HTGD = []
    ATGD = []

    # 全年一共380场比赛
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
        
    # 把每个队的 HTGS ATGS 信息补充到 dataframe 中
    playing_stat.loc[:,'HTGD'] = HTGD
    playing_stat.loc[:,'ATGD'] = ATGD

    return playing_stat

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

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2,39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    
    matchres_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return matchres_points

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
    # M 代表 unknown， 因为没有那么多历史
    h = ['M' for i in range(num * 10)]  
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num*10),380):
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
# HTGD, ATGD ,HTP, ATP的值 除以 week 数，得到平均分
cols = ['HTGD','ATGD','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW
playing_stat.tail()
playing_stat.keys()
# 抛弃前三周的比赛，抛弃队名和比赛周信息
playing_stat = playing_stat[playing_stat.MW > 3]
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
playing_stat.to_csv("final_dataset.csv")
# 读取数据
playing_stat = pd.read_csv('final_dataset.csv')
playing_stat.head()
# 删除读取
playing_stat.drop(['Unnamed: 0'],1, inplace=True)

playing_stat.head()
playing_stat.head()
# 数据分为特征和标签
X_all = playing_stat.drop(['FTR'],1)
y_all = playing_stat['FTR']

# 数据标准化
from sklearn.preprocessing import scale

cols = [['HTGD','ATGD','HTP','ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])

X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


def preprocess_features(X):
    '''把离散的类型特征转为哑编码特征 '''

    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                
        output = output.join(col_data)

    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
# 预览处理好的数据
print("\nFeature values:")
# display(X_all.head())
from sklearn.model_selection import train_test_split

# 把数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 50,
                                                    random_state = 2,
                                                    stratify = y_all)

from time import time 
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' 训练模型 '''

    # 记录训练时长
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    print("训练时间 {:.4f} 秒".format(end - start))


def predict_labels(clf, features, target):
    ''' 使用模型进行预测 '''

    # 记录预测时长
    start = time()
    y_pred = clf.predict(features)

    end = time()

    print("预测时间 in {:.4f} 秒".format(end - start))

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' 训练并评估模型 '''

    # Indicate the classifier and the training set size
    print("训练 {} 模型，样本数量 {}. . .".format(clf.__class__.__name__, len(X_train)))

    # 训练模型
    train_classifier(clf, X_train, y_train)

    # 在测试集上评估模型
    f1, acc = predict_labels(clf, X_train, y_train)
    print("训练集上的 F1 分数和准确率为: {:.4f} , {:.4f}.".format(f1 , acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("测试集上的 F1 分数和准确率为: {:.4f} , {:.4f}.".format(f1 , acc))
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 分别建立三个模型
clf_A = LogisticRegression(random_state = 42) # 线性回归
clf_B = SVC(random_state = 42, kernel='rbf',gamma='auto') # 支持向量机
clf_C = xgb.XGBClassifier(seed = 42) # xgboost

train_predict(clf_A, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print('')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb

# 设置想要自动调参的参数
parameters = { 'n_estimators':[90,100,110],
                'max_depth': [5,6,7],
                }  

# 初始化模型
clf = xgb.XGBClassifier(seed=42)

f1_scorer = make_scorer(f1_score,pos_label='H')

# 使用 grdi search 自动调参
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

grid_obj = grid_obj.fit(X_train,y_train)

# 得到最佳的模型
clf = grid_obj.best_estimator_
print(clf)

# 查看最终的模型效果
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))
import joblib
#保存模型
joblib.dump(clf, 'xgboost_model.model') 

#读取模型
xgb = joblib.load('xgboost_model.model')
sample1 = X_test.sample(n=1, random_state=1)
sample1
sample1.keys()
y_pred = xgb.predict(sample1)
y_pred
print(y_pred)
# import pandas as pd
# import joblib
# def handle(conf): 
#     """
#     该方法是部署之后，其他人调用你的服务时候的处理方法。
#     请按规范填写参数结构，这样我们就能替你自动生成配置文件，方便其他人的调用。
#     范例：
#     params['key'] = value # value_type: str # description: some description
#     参数请放到params字典中，我们会自动解析该变量。
#     """

#     param1 = conf['HTGD']  # value_type: float # description: 主场队伍本赛季本次比赛前的平均净胜球数
#     param2 = conf['ATGD']  # value_type: float # description: 客场队伍本赛季本次比赛前的平均净胜球数
#     param3 = conf['HTP']  # value_type: float # description: 主场队伍本赛季本次比赛前的平均每周得分
#     param4 = conf['ATP']  # value_type: float # description: 客场队伍本赛季本次比赛前的平均每周得分
#     param5 = conf['HM1_D']  # value_type: int # description: 主场队伍上次比赛平局与否
#     param6 = conf['HM1_L']  # value_type: int # description: 主场队伍上次比赛失败与否
#     param7 = conf['HM1_W']  # value_type: int # description: 主场队伍上次比赛胜利与否
#     param8 = conf['AM1_D']  # value_type: int # description: 客场队伍上次比赛平局与否
#     param9 = conf['AM1_L']  # value_type: int # description: 客场队伍上次比赛失败与否
#     param10 = conf['AM1_W']  # value_type: int # description: 客场队伍上次比赛胜利与否
#     param11 = conf['HM2_D']  # value_type: int # description: 主场队伍上上次比赛平局与否
#     param12 = conf['HM2_L']  # value_type: int # description: 主场队伍上上次比赛失败与否
#     param13 = conf['HM2_W']  # value_type: int # description: 主场队伍上上次比赛胜利与否
#     param14 = conf['AM2_D']  # value_type: int # description: 客场队伍上上次比赛平局与否
#     param15 = conf['AM2_L']  # value_type: int # description: 客场队伍上上次比赛失败与否
#     param16 = conf['AM2_W']  # value_type: int # description: 客场队伍上上次比赛胜利与否
#     param17 = conf['HM3_D']  # value_type: int # description: 主场队伍上上上次比赛平局与否
#     param18 = conf['HM3_L']  # value_type: int # description: 主场队伍上上上次比赛失败与否
#     param19 = conf['HM3_W']  # value_type: int # description: 主场队伍上上上次比赛胜利与否
#     param20 = conf['AM3_D']  # value_type: int # description: 客场队伍上上上次比赛平局与否
#     param21 = conf['AM3_L']  # value_type: int # description: 客场队伍上上上次比赛失败与否
#     param22 = conf['AM3_W']  # value_type: int # description: 客场队伍上上上次比赛胜利与否

#     df = pd.DataFrame.from_dict(conf)
#     model = joblib.load('xgboost_model.model')
#     result = model.predict(df)
#     return {'res': result.tolist()[0]}
# pass
# # return your result consistent with .yml you defined
# # .e.g return {'iris_class': 1, 'possibility': '88%'}


# if __name__ == '__main__':
#     conf = {}
#     handle(conf)
