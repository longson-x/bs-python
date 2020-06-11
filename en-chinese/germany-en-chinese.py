import os
import pandas as pd
import json


def getName(country):
    loc = "./en-chinese/%s/" % country

    data1 = pd.read_csv(loc + '05-06one.csv')
    data2 = pd.read_csv(loc + '05-06two.csv')
    data3 = pd.read_csv(loc + '18-19one.csv')
    data4 = pd.read_csv(loc + '18-19two.csv')

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

    print(name1)
    print(name2)
    print(name3)
    print(name4)

    cnName1 = ["拜仁慕尼黑", "比勒菲尔德", "多特蒙德", "杜伊斯堡", "法兰克福", "科隆", "汉堡", "汉诺威96", "柏林赫塔", "凯泽斯劳滕", "勒沃库森", "门兴格拉德巴赫", "美因茨", "纽伦堡", "沙尔克04", "斯图加特", "云达不莱梅", "沃尔夫斯堡"]

    cnName2 = ["亚琛", "阿伦", "波鸿", "布伦瑞克", "布格豪森", "科特布斯", "德累斯顿", "奥厄", "弗赖堡", "菲尔特", "罗斯托克", "卡尔斯鲁厄",  "慕尼黑1860", "奥芬巴查踢球者", "帕德博恩", "萨尔布吕肯", "西尔根", "温特哈兴"]

    cnName3 = ["奥格斯堡", "拜仁慕尼黑", "多特蒙德", "法兰克福", "杜塞尔多夫", "弗赖堡", "汉诺威96", "柏林赫塔", "霍芬海姆", "勒沃库森", "门兴格拉德巴赫", "美因茨", "纽伦堡", "莱比锡红牛", "沙尔克04", "斯图加特", "云达不莱梅", "沃尔夫斯堡"]

    cnName4 = ["比勒菲尔德", "波鸿", "达姆斯塔特", "德累斯顿", "杜伊斯堡", "奥厄", "科隆", "菲尔特", "汉堡", "海登海默", "荷尔斯泰因基尔", "因戈尔施塔特", "马格德堡", "帕德博恩", "雷根斯堡", "桑德豪森", "圣保利", "柏林联"]

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
    getName('germany')
