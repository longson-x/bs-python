import os
import pandas as pd
import json


def getName(country):
    loc = "./en-chinese/%s/" % country

    data1 = pd.read_csv(loc + '05-06one.csv')
    data2 = pd.read_csv(loc + '05-06two.csv')
    data3 = pd.read_csv(loc + '05-06three.csv')
    data4 = pd.read_csv(loc + '05-06four.csv')

    name1 = []
    name2 = []
    name3 = []
    name4 = []
    for i in data1.groupby('HomeTeam').mean().T.columns:
        name1.append(i)
    for i in data2.groupby('HomeTeam').mean().T.columns:
        name2.append(i)
    for i in data3.groupby('HomeTeam').mean().T.columns:
        name3.append(i)
    for i in data4.groupby('HomeTeam').mean().T.columns:
        name4.append(i)

    # print(name1)
    # print(name2)
    # print(name3)
    # print(name4)

    cnName1 = ["阿森纳", "阿斯顿维拉", "伯明翰", "布莱克本", "博尔顿", "查尔顿", "切尔西", "埃弗顿", "富勒姆",
               "利物浦", "曼城", "曼联", "米德尔斯堡", "纽卡斯尔", "朴次茅斯", "桑德兰", "热刺", "西布罗姆维奇", "西汉姆联", "维冈竞技"]
    cnName2 = ["布莱顿", "伯恩利", "卡迪夫城", "考文垂", "克鲁", "水晶宫", "德比郡", "赫尔城", "伊普斯威奇", "利兹联", "莱斯特城", "卢顿",
               "米尔沃尔", "诺维奇", "普利茅斯", "普雷斯顿", "女王公园巡游者", "雷丁", "谢菲尔德联", "谢菲尔德周三", "南安普敦", "斯托克城", "沃特福德", "狼队"]
    cnName3 = ["巴恩斯利", "布莱克浦", "伯恩茅斯", "布拉德福德", "布伦特福德", "布里斯托城", "切斯特菲尔德", "科切斯特联", "唐卡斯特", "吉林汉姆", "哈特利浦",
               "哈德斯菲尔德", "米尔顿凯恩斯", "诺丁汉森林", "奥德汉姆", "维尔港", "罗瑟汉姆", "斯坎索普", "南安联", "斯旺西", "斯文登", "特兰米尔", "沃尔索尔", "约维尔"]
    cnName4 = ["巴尼特", "波士顿联队", "布里斯托流浪", "伯利", "卡利斯尔联", "切尔滕纳姆", "切斯特", "达宁顿", "格林斯比", "莱顿东方", "林肯城", "马科斯菲尔德",
               "曼斯菲尔德", "北安普敦", "诺茨郡", "牛津联队", "彼得堡联", "罗奇代尔", "鲁什登钻石", "谢斯伯利", "斯托克港", "托奎联", "雷克斯汉姆", "韦康比流浪者"]

    all = []
    for i in range(len(name1)):
        all.append({
            "name": name1[i],
            "cnName": cnName1[i]
        })
    for i in range(len(name2)):
        all.append({
            "name": name2[i],
            "cnName": cnName2[i]
        })
    for i in range(len(name3)):
        all.append({
            "name": name3[i],
            "cnName": cnName3[i]
        })
    for i in range(len(name4)):
        all.append({
            "name": name4[i],
            "cnName": cnName4[i]
        })
    # print(all)
    if not os.path.exists('./en-chinese/%s'%country):
        os.makedirs('./en-chinese/%s'%country)
    with open('./en-chinese/%s/%s_cname.json'%(country,country), 'w', encoding='utf-8') as f:
        json.dump(all, f, ensure_ascii=False)


if __name__ == '__main__':
    getName('italy')
