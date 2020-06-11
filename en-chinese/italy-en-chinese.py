import os
import pandas as pd
import json


def getName(country):
    loc = "./en-chinese/%s/" % country

    data1 = pd.read_csv(loc + '06-07one.csv')
    data2 = pd.read_csv(loc + '06-07two.csv')
    data3 = pd.read_csv(loc + '12-13one.csv')
    data4 = pd.read_csv(loc + '12-13two.csv')
    data5 = pd.read_csv(loc + '18-19one.csv')

    name1 = []
    name2 = []
    name3 = []
    name4 = []
    name5 = []
    for i in data1.groupby('HomeTeam').mean().T.columns:
        name1.append(i)
    for i in data2.groupby('HomeTeam').mean().T.columns:
        name2.append(i)
    for i in data3.groupby('HomeTeam').mean().T.columns:
        name3.append(i)
    for i in data4.groupby('HomeTeam').mean().T.columns:
        name4.append(i)
    for i in data5.groupby('HomeTeam').mean().T.columns:
        name5.append(i)

    # print(name1)
    # print(name2)
    # print(name3)
    # print(name4)
    # print(name5)

    cnName1 = ["阿斯科利", "亚特兰大", "卡利亚里", "卡塔尼亚",  "切沃", "恩波利", "佛罗伦萨", "国际米兰", "拉齐奥",
               "利沃诺", "梅西纳", "AC米兰", "巴勒莫", "帕尔马", "雷吉纳", "罗马", "桑普多利亚", "锡耶纳", "都灵", "乌迪内斯"]

    cnName2 = ["阿尔比诺勒菲", '阿雷佐', "巴里",  "博洛尼亚", "布雷西亚", "切塞纳", "克罗托内", "弗洛西诺尼", "热那亚", "尤文图斯",
               "莱切", "曼托瓦", "摩德纳", "那不勒斯", "佩斯卡拉", "皮亚琴察", "里米尼", "斯佩齐亚", "特雷维索", "特里埃斯蒂纳", "维罗纳", "维琴察"]

    cnName3 = ["亚特兰大", "博洛尼亚", "卡利亚里", "卡塔尼亚",  "切沃", "佛罗伦萨", "热那亚", "国际米兰", "尤文图斯",
               "拉齐奥", "AC米兰", "那不勒斯", "巴勒莫", "帕尔马", "佩斯卡拉", "罗马", "桑普多利亚", "锡耶纳", "都灵", "乌迪内斯"]

    cnName4 = ["阿斯科利", "巴里", "布雷西亚", "切塞纳", "希塔德拉", "克罗托内", "恩波利", "格罗瑟托", "史泰比亚", "利沃诺",
               "摩德纳", "诺瓦拉", "帕多瓦", "韦尔切利", "雷吉纳", "萨索洛", "斯佩齐亚", "特拉纳", "瓦雷斯", "维罗纳", "维琴察", "兰西安奴"]

    cnName5 = ["亚特兰大", "博洛尼亚", "卡利亚里", "切沃", "恩波利", "佛罗伦萨", "弗洛西诺尼", "热那亚", "国际米兰", "尤文图斯",
               "拉齐奥", "AC米兰", "那不勒斯", "帕尔马", "罗马", "桑普多利亚", "萨索洛", "斯帕尔", "都灵", "乌迪内斯"]

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
    for i in range(len(name5)):
        all.append({
            "name": name5[i],
            "cnName": cnName5[i]
        })
    # print(all)
    # print(len(all))

    length = len(all)
    remove = []
    for i in range(length):
        for j in range(length):
            if (all[i]['name'] == all[j]['name']) & (all[i]['cnName'] == all[j]['cnName']) & (i < j):
                remove.append(j)
    # print(remove)

    newAll = []
    for i in range(length):
        if i not in remove:
            newAll.append(all[i])
    # print(newAll)

    if not os.path.exists('./en-chinese/%s' % country):
        os.makedirs('./en-chinese/%s' % country)
    with open('./en-chinese/%s/%s_cname.json' % (country, country), 'w', encoding='utf-8') as f:
        json.dump(newAll, f, ensure_ascii=False)


if __name__ == '__main__':
    getName('italy')
