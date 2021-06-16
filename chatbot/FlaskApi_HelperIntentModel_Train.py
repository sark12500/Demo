#!flask/bin/python
# -*- coding: utf-8 -*-

import sys

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

from flask_cors import CORS
# from gevent.pywsgi import WSGIServer
import pandas as pd
import numpy as np

from FlaskApi_HelperIntentModel_Sentence import *
from FlaskApi_HelperIntentModel_Train_Log import *
from FlaskApi_CheckAttribute import *
from FlaskApi_Check_HelperIntentModel import *
from HelperIntentModelUtility import HelperIntentModelUtility
from HelperDataUtility_Train import HelperDataUtility
from StateCode import StateCode
from Utility_Logger import UtilityLogger
from NLP_IntentModelFactory import *
from NLP_JiebaSegmentor import Get_JiebaSegmentor
from MqDAO import MqDAO
from MyEnv import Get_MyEnv

from tensorflow import Graph, Session
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import scikitplot as skplt

app = Flask(__name__)
CORS(app)


"""
train at remote VM
"""


class IntentModelTrainingRemoteView(MethodView):

    @requires_helper_intent_model_training_type
    @requires_helper_intent_model_training
    @requires_json
    def post(self):
        """
        model訓練流程：
        1.檢查json參數
        2.檢查是否符合訓練條件 EX：robot, 演算法， 語句集......是否存在
        3.建立演算法工廠(分爲深度學習與機器學習, 各演算法需要的特徵工程與參數不一樣)
        4.依照條件進行隨機切割出測試資料
        5.將訓練預測資料進行前處理and特徵工程
        6.新增訓練紀錄(train_log), 狀態爲 TRAIN 訓練中
        7.訓練模型
        8.將測試資料丟入模型驗證並產生相關評估統計值
        9.實體模型and統計圖表存到fs
        10.將統計值存入DB(test_log), 更改訓練紀錄(train_log)狀態爲 N 未上線
        11.將此次訓練語句集存入DB(sentence_log)
        12.整理部分統計值回給前端


        input
        {
            "robot_id":"hr.00001318",
            "model_id":"model_name",
            "sentence_set_id": "sentence_01",
            "algorithm":"textcnn",
            "modify_user":"charles.huang@quantatw.com",
            "train_test_split_size": 0.8
            "sentence_max_len": 20
        }
        """

        # try:

        robot_id = request.json['robot_id']
        model_id = request.json['model_id']
        sentence_set_id = request.json['sentence_set_id']
        algorithm = request.json['algorithm']
        modify_user = request.json['modify_user']

        """
        optional 調整機會才開放出來
        """
        # 向量長度 - 取最長的單字向量長度
        sentence_max_len = request.json.get('sentence_max_len', 10)
        # 訓練測試切割
        train_test_split_size = request.json.get('train_test_split_size', 0.8)

        df = HelperIntentModelUtility.query_model(robot_id=robot_id)
        if len(df) == 0:
            return jsonify(code=StateCode_HelperData.RobotNotExist,
                           message="This robot_id={} is not exist.".format(robot_id),
                           data=[],
                           )

        # # 使用Queue, 可以一次發出多個訓練request去排隊, 只要model_id不同
        # # 一個機器人只能同時train一個model
        # df = HelperIntentModelUtility.query_train_log(robot_id=robot_id)
        # if len(df[df.status == 'TRAIN']) > 0 or len(df[df.status == 'QUEUE']) > 0:
        #     return jsonify(code=StateCode_HelperData.ModelInTraining,
        #                    message="This robot_id={} is training.".format(robot_id),
        #                    data=[],
        #                    )

        # model已經存在
        df = HelperIntentModelUtility.query_train_log(robot_id=robot_id,
                                                      model_id=model_id)
        if len(df[df.model_id == model_id]) > 0:
            return jsonify(code=StateCode_HelperData.ModelExist,
                           message="This robot_id={} + model_id={} has existed.".format(robot_id, model_id),
                           data=[],
                           )

        df = HelperIntentModelUtility.query_train_sentence_set(robot_id=robot_id,
                                                               sentence_set_id=sentence_set_id)
        if len(df) == 0:
            return jsonify(code=StateCode_HelperData.SentenceSetNotExist,
                           message="This robot_id={} + sentence_set_id={} has not existed.".format(robot_id,
                                                                                                   sentence_set_id),
                           data=[],
                           )

        df = HelperIntentModelUtility.query_algorithm(algorithm_id=algorithm)
        if len(df) == 0:
            return jsonify(code=StateCode_HelperData.AlgorithmNotExist,
                           message="This robot_id={} + algorithm={} has not existed.".format(robot_id,
                                                                                             algorithm),
                           data=[],
                           )

        algorithm_type = df['algorithm_type'][0]

        """
        判斷方法1: robot的上傳語句集要包含2個skill以上 + 每個skill句子數量 > 5 才能train
        EX：robot有5個skill, 語句集01 只有2個skill也可以train, 但這2個skill句數都須>5

        判斷方法2: robot的每個skill都要大於5句

        目前採用方法1
        """
        sentence_df = HelperIntentModelUtility.query_train_sentence(robot_id=robot_id,
                                                                    sentence_set_id=sentence_set_id)

        # 不存在的技能的不能上傳
        upload_skill_list = list(set(sentence_df.skill_id))
        robot_skill_df = HelperIntentModelUtility.query_robot_skill(robot_id)
        robot_skill_list = list(robot_skill_df.skill_id)
        for upload_skill in upload_skill_list:

            if upload_skill not in robot_skill_list:
                data = dict(
                    robot_id=robot_id,
                    sentence_set_id=sentence_set_id,
                    robot_skill_list=robot_skill_list,
                    upload_skill_list=upload_skill_list
                )

                return jsonify(code=StateCode_HelperData.SkillNotExist,
                               data=[data],
                               message='The file has wrong format. skill is not in robot. robot_id={robot_id}'.format(
                                   robot_id=robot_id)
                               )

        train_sentence_skill_sum = sentence_df.groupby(["skill_id"]).size()

        # 每個skill句子數量 > 5
        each_skill_count = len(train_sentence_skill_sum[train_sentence_skill_sum < 5])
        if each_skill_count > 0:
            return jsonify(code=StateCode_HelperData.SentenceNotEnough,
                           message=("This robot_id={} + sentence_set_id={} " +
                                    "every skill_id count in sentence_set must >= 5 .").format(
                               robot_id,
                               sentence_set_id),
                           data=[],
                           )
        # 語句集要 1個 skill以上才能train
        if len(train_sentence_skill_sum) < 1:
            return jsonify(code=StateCode_HelperData.SkillNotEnough,
                           message=("This robot_id={} + sentence_set_id={} " +
                                    "skill_id count  must >=1  .").format(
                               robot_id,
                               sentence_set_id),
                           data=[],
                           )

        # 新增訓練紀錄 status = QUEUE
        HelperIntentModelUtility.insert_train_log(robot_id=robot_id,
                                                  model_id=model_id,
                                                  sentence_set_id=sentence_set_id,
                                                  algorithm=algorithm,
                                                  modify_user=modify_user,
                                                  algorithm_param='',
                                                  algorithm_type='',
                                                  tokenizer_id='',
                                                  mapping='',
                                                  mapping_name='',
                                                  train_test_size=train_test_split_size,
                                                  status="QUEUE",
                                                  nfs_model_id='',
                                                  nfs_tokenizer_id=''
                                                  )

        # publish 到mq去排隊訓練
        if algorithm_type == 'ML':
            queue_id = Get_MyEnv().env_mq_name_intent_train_ml_queue
        elif algorithm_type == 'DL':
            queue_id = Get_MyEnv().env_mq_name_intent_train_dl_queue
        elif algorithm_type == 'BERT':
            queue_id = Get_MyEnv().env_mq_name_intent_train_bert_queue
        else:
            return jsonify(code=StateCode_HelperData.ModelNotExist,
                           data=[],
                           message='algorithm_type is not exist'
                           ), 999

        logger.info('queue_id : {}'.format(queue_id))
        body = {
            "robot_id": robot_id,
            "model_id": model_id,
            "sentence_set_id": sentence_set_id,
            "algorithm": algorithm,
            "modify_user": modify_user,
            "train_test_split_size": train_test_split_size,
            "sentence_max_len": sentence_max_len
        }

        mq.publish(queue_id, body)

        return jsonify(code=StateCode.Success,
                       message="success",
                       data=[]
                       )

        # except Exception as e:
        #
        #     utility_logger = UtilityLogger()
        #     msg = utility_logger.except_error_msg(sys.exc_info())
        #     logger.error(msg)
        #     log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
        #     utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    data=[],
        #                    message=msg
        #                    ), 999


def get_model(robot_id=None):
    logger.debug('get_model')
    pd_df = HelperIntentModelUtility.query_model(robot_id=robot_id)

    create_user_list = list(pd_df.create_user.unique())
    modify_user_list = list(pd_df.modify_user.unique())
    # 重覆user過濾掉
    all_user_list = list(set(create_user_list + modify_user_list))
    # 先將所有owner資料查出來
    user_info_dict = {}

    # 一次全撈
    info_list = HelperDataUtility.get_user(_camp_id=all_user_list)
    for info in info_list:
        user_info_dict.update({info['campUserId']: info})

    data = []
    for index, row in pd_df.iterrows():
        result = dict(
            robotId=row['robot_id'],
            minConfidence=row['min_confidence'],
            create_user=user_info_dict.get(row['create_user'], {}),
            createDate=row['create_date'],
            modify_user=user_info_dict.get(row['modify_user'], {}),
            modifyDate=row['modify_date']
        )

        data.append(result)
    logger.debug('data')
    logger.debug(data)
    logger.debug('len(data) = ' + str(len(data)))
    return data


if __name__ == "__main__":

    # log
    # 系統log只顯示error級別以上的
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    # 自訂log
    logger = logging.getLogger('FlaskAPi_HelperIntentModel_Train.py')
    logger.setLevel(logging.DEBUG)

    config = Get_HelperConfig()

    MSSQL_DB = Get_MyEnv().env_mssql_db

    HELPER_ERROR_LOG_TABLE = config.HELPER_ERROR_LOG_TABLE
    HELPER_INTENT_MODEL_TABLE = config.HELPER_INTENT_MODEL_TABLE
    HELPER_INTENT_TRAIN_SENTENCE_TABLE = config.HELPER_INTENT_TRAIN_SENTENCE_TABLE
    HELPER_INTENT_TRAIN_LOG_TABLE = config.HELPER_INTENT_TRAIN_LOG_TABLE
    HELPER_INTENT_TEST_LOG_TABLE = config.HELPER_INTENT_TEST_LOG_TABLE

    config = Get_FormatConfig()
    DATE_FORMAT_NORMAL = config.DATE_FORMAT_NORMAL

    mq = MqDAO(url=Get_MyEnv().env_mq_url,
               port=Get_MyEnv().env_mq_port,
               account=Get_MyEnv().env_mq_account,
               password=Get_MyEnv().env_mq_password)

    # init jieba
    jieba = Get_JiebaSegmentor()

    app.add_url_rule('/api/helper/intentModelTraining',
                     view_func=IntentModelTrainingView.as_view('IntentModelTrainingView'))

    app.add_url_rule('/api/helper/intentModelTrainingRemote',
                     view_func=IntentModelTrainingRemoteView.as_view('IntentModelTrainingRemoteView'))

    logger.info('api run !!')

    if Get_MyEnv().env_oo == Get_MyEnv().LOCAL:
        app.run(port=5019, host='127.0.0.1', debug=False, use_reloader=False, threaded=True)
    else:
        app.run(port=5019, host='0.0.0.0', debug=True, use_reloader=False, threaded=True)

    # HelperData 5005
    # SparkSubmit_HelperData 5006
    # HelperDataModel 5007
    # HelperData_mssql 5010
    # HelperDataModel_mssql 5011
    # HelperDataModel_mssql_py2 5019
    # HelperDataModel_mssql_py3 5020
