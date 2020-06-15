import pandas as pd
import joblib
# from pymongo import MongoClient as Client

from flask import Flask, request
import json
server = Flask(__name__)  #__name__代表当前的python文件。把当前的python文件当做一个服务启动


@server.route('/api/getResult',methods=['post'])  #只有在函数前加上@server.route (),这个函数才是个接口，不是一般的函数
def prediction():
    # conf = request.form.get('conf')
    country = request.values.get('country')
    conf = request.values.get('conf')
    # print(json.loads(conf))
    # print(json.loads(conf)['input'])
    # 判断入参是否为空
    print(conf)
    if conf:
        df = pd.DataFrame.from_dict(json.loads(conf), orient='index')
        model = joblib.load('./%s_xgboost_model.model'%country)
        result = model.predict(df)
        return {'code': 0, 'msg': '使用模型预测成功', 'data': result.tolist()[0]}
        # return conf
    else:
        return {'code': 1001, 'msg': '调用接口失败', 'data': ''}  #1001表示必填接口未填


@server.route('/api/getSeasonData', methods=['post'])
def getSeasonData():
    country = request.values.get('country')
    season = request.values.get('season')
    if season:
        # json文件形式返回数据

        length = 0
        with open('./files/%s/%s.json' % (country, season),'r',encoding='utf-8') as fp:
            allData = json.load(fp)
        for i in allData:
            length = length + 1
        return {'code': 0, 'msg': '数据请求成功', 'data': allData[length-1]}

        # myclient = Client('mongodb://localhost:27017/')
        # mydb = myclient['bs_db']
        # mycol = mydb[country]
        # countQuery = {'season': season}
        # count = mycol.find(countQuery).count()
        # dataQuery = {'season': season, 'week': str(count)}
        # data = {}
        # for item in mycol.find(dataQuery):
        #     item.pop('_id')
        #     data = item
        # return {'code': 0, 'msg': '数据请求成功', 'data': data}
    else:
        return {'code': 1001, 'msg': '调用接口失败', 'data': ''}


@server.route('/api/getSeasonCurrentData', methods=['post'])
def getSeasonCurrentData():
    country = request.values.get('country')
    season = request.values.get('season')
    num = request.values.get('num')
    if num:
        # json文件形式返回数据

        with open('./files/%s/%s.json' % (country, season),'r',encoding='utf-8') as fp:
            allData = json.load(fp)
        return {'code': 0, 'msg': '数据请求成功', 'data': allData[int(num) - 1]}

        # myclient = Client('mongodb://localhost:27017/')
        # mydb = myclient['bs_db']
        # mycol = mydb[country]
        # dataQuery = {'season': season, 'week': num}
        # data = {}
        # for item in mycol.find(dataQuery):
        #     item.pop('_id')
        #     data = item
        # return {'code': 0, 'msg': '数据请求成功', 'data': data}
    else:
        return {'code': 1001, 'msg': '调用接口失败', 'data': ''}

@server.route('/api/getSeasonSchedule', methods=['post'])
def getSeasonSchedule():
    country = request.values.get('country')
    if country:
        with open('./files/%s/%s_2019-20_schedule.json' % (country, country),'r',encoding='utf-8') as fp:
            allData = json.load(fp)
        return {'code': 0, 'msg': '数据请求成功', 'data': allData}
    else:
        return {'code': 1001, 'msg': '调用接口失败', 'data': ''}

server.run(port=5000, debug=True, host='127.0.0.1')
#端口不写默认是5000.debug=True表示改了代码后不用重启，会自动帮你重启.host写0.0.0.0，别人就可以通过ip访问接口。否则就是127.0.0.1
