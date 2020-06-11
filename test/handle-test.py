import pandas as pd
import joblib
def handle(conf, country): 
    """
    该方法是部署之后，其他人调用你的服务时候的处理方法。
    请按规范填写参数结构，这样我们就能替你自动生成配置文件，方便其他人的调用。
    范例：
    params["key"] = value # value_type: str # description: some description
    参数请放到params字典中，我们会自动解析该变量。
    """

    # param1 = conf["HTGD"]  # value_type: float # description: 主场队伍本赛季本次比赛前的平均净胜球数
    # param2 = conf["ATGD"]  # value_type: float # description: 客场队伍本赛季本次比赛前的平均净胜球数
    # param3 = conf["HTP"]  # value_type: float # description: 主场队伍本赛季本次比赛前的平均每周得分
    # param4 = conf["ATP"]  # value_type: float # description: 客场队伍本赛季本次比赛前的平均每周得分
    # param5 = conf["HM1_D"]  # value_type: int # description: 主场队伍上次比赛平局与否
    # param6 = conf["HM1_L"]  # value_type: int # description: 主场队伍上次比赛失败与否
    # param7 = conf["HM1_W"]  # value_type: int # description: 主场队伍上次比赛胜利与否
    # param8 = conf["AM1_D"]  # value_type: int # description: 客场队伍上次比赛平局与否
    # param9 = conf["AM1_L"]  # value_type: int # description: 客场队伍上次比赛失败与否
    # param10 = conf["AM1_W"]  # value_type: int # description: 客场队伍上次比赛胜利与否
    # param11 = conf["HM2_D"]  # value_type: int # description: 主场队伍上上次比赛平局与否
    # param12 = conf["HM2_L"]  # value_type: int # description: 主场队伍上上次比赛失败与否
    # param13 = conf["HM2_W"]  # value_type: int # description: 主场队伍上上次比赛胜利与否
    # param14 = conf["AM2_D"]  # value_type: int # description: 客场队伍上上次比赛平局与否
    # param15 = conf["AM2_L"]  # value_type: int # description: 客场队伍上上次比赛失败与否
    # param16 = conf["AM2_W"]  # value_type: int # description: 客场队伍上上次比赛胜利与否
    # param17 = conf["HM3_D"]  # value_type: int # description: 主场队伍上上上次比赛平局与否
    # param18 = conf["HM3_L"]  # value_type: int # description: 主场队伍上上上次比赛失败与否
    # param19 = conf["HM3_W"]  # value_type: int # description: 主场队伍上上上次比赛胜利与否
    # param20 = conf["AM3_D"]  # value_type: int # description: 客场队伍上上上次比赛平局与否
    # param21 = conf["AM3_L"]  # value_type: int # description: 客场队伍上上上次比赛失败与否
    # param22 = conf["AM3_W"]  # value_type: int # description: 客场队伍上上上次比赛胜利与否

    df = pd.DataFrame.from_dict(conf, orient="index")
    model = joblib.load("%s_xgboost_model.model"%country)
    result = model.predict(df)
    # return {"res": result.tolist()[0]}
    return result
# pass
# return your result consistent with .yml you defined
# .e.g return {"iris_class": 1, "possibility": "88%"}


if __name__ == "__main__":
    conf = {
        "input": {
        "HTGD": 3.5,
        "ATGD": 6.2,
        "HTP": 2.1,
        "ATP": 1.3,
        "HM1_D": 0,
        "HM1_L": 0,
        "HM1_W": 1,
        "AM1_D": 0,
        "AM1_L": 1,
        "AM1_W": 0,
        "HM2_D": 0,
        "HM2_L": 0,
        "HM2_W": 1,
        "AM2_D": 0,
        "AM2_L": 1,
        "AM2_W": 0,
        "HM3_D": 0,
        "HM3_L": 0,
        "HM3_W": 1,
        "AM3_D": 0,
        "AM3_L": 1,
        "AM3_W": 0
        }
    }
    country = 'england'
    result = handle(conf,country)
    print(result)
    # if (result["res"] == "H"):
    #     print("主队胜")
    