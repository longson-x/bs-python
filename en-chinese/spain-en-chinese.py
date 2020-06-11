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

    # print(name1)
    # print(name2)
    # print(name3)
    # print(name4)

    cnName1 = ["阿拉维斯", "毕尔巴鄂竞技", "马德里竞技", "巴塞罗那", "贝蒂斯", "卡迪斯", "塞尔塔", "西班牙人", "赫塔菲", "拉科鲁尼亚", "马拉加", "马洛卡", "奥萨苏纳", "皇家马德里", "桑坦德竞技", "塞维利亚", "皇家社会", "瓦伦西亚", "比利亚雷亚尔", "萨拉戈萨",]

    cnName2 = ["阿尔巴塞特", "阿尔梅里亚", "卡斯迪隆", "穆尔西亚", "埃瓦尔", "埃尔切", "费罗尔竞技", "塔拉戈纳", "赫库斯", "莱万特", "莱里达", "洛卡", "马拉加B队", "穆尔西亚", "努曼西亚", "艾积多", "皇家马德里B队", "维尔瓦", "希洪竞技", "特内里费", "巴利亚多利德", "萨雷斯"]

    cnName3 = ["阿拉维斯", "毕尔巴鄂竞技", "马德里竞技", "巴塞罗那", "贝蒂斯", "塞尔塔", "埃瓦尔", "西班牙人", "赫塔菲", "赫罗纳", "韦斯卡", "莱加内斯", "莱万特", "皇家马德里", "塞维利亚", "皇家社会", "瓦伦西亚", "巴利亚多利德", "巴列卡诺", "比利亚雷亚尔"]

    cnName4 = ["阿尔巴塞特", "艾科坎", "阿尔梅里亚", "卡迪斯", "科尔多瓦", "埃尔切", "埃斯特雷马杜拉", "费罗尔竞技", "格拉纳达", "拉科鲁尼亚", "拉斯帕尔马斯", "鲁勾", "马拉加", "马洛卡", "努曼西亚", "奥萨苏纳", "奥维耶多", "马哈达洪达", "雷乌斯", "希洪竞技", "特内里费", "萨拉戈萨"]

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
    getName('spain')
