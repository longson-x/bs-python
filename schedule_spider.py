import urllib.request
import urllib.parse
import json
import time
import threadpool
import os

def getScore(league):
    tab = '赛程'
    word_league = urllib.parse.quote(league)
    word_tab = urllib.parse.quote(tab)
    url ='https://dc.qiumibao.com/shuju/public/index.php?_url=/data/index&league=%s&tab=%s&year=[year]'%(word_league, word_tab)
    res = urllib.request.urlopen(url)
    html_data = res.read().decode('utf-8')

    # 将python对象test转换json对象
    # data = json.dumps(html_data, ensure_ascii=False)
    # print(data)

    # 将json对象转换成python对象
    load = json.loads(html_data)
    a = ["timestamp", "date", "dateTime", "homeId", "guestId", "homeTeam", "awayTeam", "score", "htImg", "atImg", "insidePage", "matchSituation", "is_finish", "htName", "atName"]
    for item in load["data"]:
        item.pop('内页')
        newList = []
        for newItem in item['list']:
            obj = {}
            num = 0
            for i in newItem:
                obj[a[num]] = newItem[i]
                num = num + 1
            obj.pop('timestamp')
            obj.pop('insidePage')
            obj.pop('matchSituation')
            newList.append(obj)
        item['list'] = newList
    # print(load["data"])

    if not os.path.exists('./files/%s' % changeName(league)):
        os.makedirs('./files/%s' % changeName(league))
    with open('./files/%s/%s_2019-20_schedule.json' % (changeName(league), changeName(league)), 'w', encoding='utf-8') as f:
        json.dump(load["data"], f, ensure_ascii=False)

def changeName(name):
    if name == '英超':
        return 'england'
    elif name == '意甲':
        return 'italy'
    elif name == '西甲':
        return 'spain'
    elif name == '德甲':
        return 'germany'
    elif name == '法甲':
        return 'france'
    else:
        return ''

start_time = time.time()
all_league = ["英超", "意甲", "西甲", "德甲", "法甲"]
task_pool = threadpool.ThreadPool(5)
requests = threadpool.makeRequests(getScore, all_league)
for req in requests:
    task_pool.putRequest(req)
    task_pool.wait()
    end = time.time()

print (end - start_time)
start1 = time.time()
for league in all_league:
    getScore(league)
print (time.time()-start1)