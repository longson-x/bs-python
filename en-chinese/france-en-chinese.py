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

    cnName1 = ["阿雅克肖", "欧塞尔", "波尔多", "勒芒", "朗斯", "里尔", "里昂", "马赛", "梅斯", "摩纳哥", "南锡", "南特", "尼斯", "巴黎圣日尔曼", "雷恩", "索肖", "圣埃蒂安", "斯特拉斯堡", "图卢兹", "特鲁瓦"]

    cnName2 = ["亚眠", "巴斯蒂亚", "布雷斯特", "卡昂", "沙特鲁", "克莱蒙", "基迪尔", "第戎", "格勒诺布尔", "格尼翁", "甘冈", "伊斯特尔", "拉瓦勒", "勒阿弗尔", "洛里昂", "蒙彼利埃", "兰斯", "色当", "席特", "瓦朗谢纳"]

    cnName3 = ["亚眠", "安格斯", "波尔多", "卡昂", "第戎", "甘冈", "里尔", "里昂", "马赛", "摩纳哥", "蒙彼利埃", "南特", "尼斯", "尼姆", "巴黎圣日尔曼", "兰斯", "雷恩", "圣埃蒂安", "斯特拉斯堡", "图卢兹"]

    cnName4 = ["阿雅克肖", "GFC阿雅克肖", "欧塞尔", "贝兹尔", "布雷斯特", "沙特鲁", "克莱蒙", "格勒诺布尔", "勒阿弗尔", "朗斯", "洛里昂", "梅斯", "南锡", "尼奥特", "奥兰斯", "巴黎足球", "红星", "索肖", "特鲁瓦", "瓦朗谢纳"]

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
    getName('france')
