#!flask/bin/python
# -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from flask_cors import CORS
# from gevent.pywsgi import WSGIServer

# from envCassandraDAO import envCassandraDAO
from MyEnv import Get_MyEnv
from Config_Format import Get_FormatConfig
from FlaskApi_HelperIntentModel_Sentence import *
from FlaskApi_HelperIntentModel_Train_Log import *
from FlaskApi_CheckAttribute import *
from FlaskApi_Check_HelperIntentModel import *
from HelperIntentModelUtility import HelperIntentModelUtility
from HelperDataUtility import HelperDataUtility
from StateCode import StateCode
from Utility_Logger import UtilityLogger
from NLP_IntentModelFactory import *
from NLP_IntentModelPredictor_UI import Get_IntentModelPredictTest
from NLP_JiebaSegmentor import Get_JiebaSegmentor

from tensorflow import Graph, Session
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import scikitplot as skplt

# GraphQL
from flask_graphql import GraphQLView
from graphene import ObjectType, String, Schema, Int, Boolean, List, Field

app = Flask(__name__)
CORS(app)


class UserRobotView(MethodView):

    @requires_helper_user_robot
    def get(self):
        """
        查詢 model
        api/helper/user_robot/charles.huang@quantatw.com
        """

        camp_user_id = request.args.get('camp_user_id')

        try:
            data = []

            logger.debug('query_user_robot start')
            time = datetime.strftime(datetime.now(), DATE_FORMAT_NORMAL)
            logger.debug(time)

            pd_df = HelperIntentModelUtility.query_user_robot(camp_user_id=camp_user_id)

            logger.debug('query_user_robot back')
            time = datetime.strftime(datetime.now(), DATE_FORMAT_NORMAL)
            logger.debug(time)
            logger.debug(pd_df)

            if any(pd_df):
                create_user_list = list(pd_df['last_train_user'].unique())
                # 重覆user過濾掉
                create_user_list = [x for x in create_user_list if x]
                all_user_list = list(set(create_user_list))

                # 先將所有owner資料查出來
                user_info_dict = {}

                # 一次全撈
                info_list = HelperDataUtility.get_user(_camp_id=all_user_list)
                for info in info_list:
                    user_info_dict.update({info['camp_user_id']: info})

                for index, row in pd_df.iterrows():
                    result = dict(
                        robot_id=row['robot_id'],
                        robot_name=row['robot_name'],
                        robot_img=row['robot_img'],
                        skill_count=row['skill_count'],
                        last_train_date=row['last_train_date'],
                        last_train_user=user_info_dict.get(row['last_train_user'], {})
                    )

                    data.append(result)

            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999


class IntentModelView(MethodView):

    @requires_helper_intent_model
    def get(self):
        """
        查詢 model
        api/helper/intentModel?robot_id=pepper.00001318
        """

        robot_id = request.args.get('robot_id')

        try:

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
                user_info_dict.update({info['camp_user_id']: info})

            data = []
            for index, row in pd_df.iterrows():
                result = dict(
                    robot_id=row['robot_id'],
                    min_confidence=row['min_confidence'],
                    new_model_id=row['new_model_id'],
                    create_user=user_info_dict.get(row['create_user'], {}),
                    create_date=row['create_date'],
                    modify_user=user_info_dict.get(row['modify_user'], {}),
                    modify_date=row['modify_date']
                )

                data.append(result)

            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999

    @requires_helper_intent_model_type
    @requires_helper_intent_model
    @requires_json
    def post(self):
        """
        插入模型
        {
            "robot_id":"pepper.00001318",
            "min_confidence":0.9 ,
            "modify_user": "charles.huang@quantatw.com",
            "new_model_id": "new_model_id"
        }
        """

        robot_id = request.json['robot_id']
        min_confidence = request.json['min_confidence']
        new_model_id = request.json['new_model_id']
        modify_user = request.json['modify_user']

        try:

            df = HelperIntentModelUtility.query_model(robot_id=robot_id)

            if len(df) > 0:
                return jsonify(code=StateCode_HelperData.RobotExist,
                               message="This robot_id={} has existed.".format(robot_id),
                               data=[],
                               )

            # # 新增model 不允許爲Y (上線)
            # if status is None or status == 'Y':
            #     return jsonify(code=StateCode.InputTypeError,
            #                    message="This robot_id={} + model_id={} status must to be N.".format(robot_id,
            #                                                                                         model_id),
            #                    data=[],
            #                    )

            # if has_new_model is None or has_new_model == True:
            #     return jsonify(code=StateCode.InputTypeError,
            #                    message="This robot_id={} + model_id={} has_new_model must to be False.".format(robot_id,
            #                                                                                                    model_id),
            #                    data=[],
            #                    )

            HelperIntentModelUtility.insert_model(robot_id=robot_id,
                                                  min_confidence=min_confidence,
                                                  new_model_id=new_model_id,
                                                  modify_user=modify_user)

            data = [request.json]
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999

    @requires_helper_intent_model_type
    @requires_helper_intent_model
    @requires_json
    def put(self):
        """
         更新模型
         {
             "robot_id":"pepper.00001318",
             "min_confidence":0.9,
             "new_model_id": False,
             "modify_user": "charles.huang@quantatw.com"
         }
         """

        robot_id = request.json['robot_id']
        min_confidence = request.json['min_confidence']
        new_model_id = request.json['new_model_id']
        modify_user = request.json['modify_user']

        try:

            df = HelperIntentModelUtility.query_model(robot_id=robot_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.RobotNotExist,
                               message="This robot_id:{} has not existed.".format(robot_id),
                               data=[],
                               )

            HelperIntentModelUtility.update_model(robot_id=robot_id,
                                                  min_confidence=min_confidence,
                                                  new_model_id=new_model_id,
                                                  modify_user=modify_user)

            data = [request.json]
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999

    @requires_helper_intent_model_type
    @requires_helper_intent_model
    @requires_json
    def patch(self):
        """
         更新模型, 可以只更新部分欄位
         {
             "robot_id":"pepper.00001318",
             "min_confidence":0.9,
             "new_model_id": new_model_id,
             "modify_user": "charles.huang@quantatw.com"
         }
         """

        json_data = request.get_json()
        logger.debug("update_model_by_column  json_data:")
        logger.debug(json_data)

        robot_id = request.json['robot_id']
        modify_user = request.json['modify_user']

        # optional
        min_confidence = json_data.get('min_confidence', None)
        new_model_id = json_data.get('new_model_id', None)

        try:

            df = HelperIntentModelUtility.query_model(robot_id=robot_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.RobotNotExist,
                               message="This robot_id={} has not existed.".format(robot_id),
                               data=[],
                               )

            # 更新部分欄位
            HelperIntentModelUtility.update_model_by_column(robot_id=robot_id,
                                                            min_confidence=min_confidence,
                                                            new_model_id=new_model_id,
                                                            modify_user=modify_user)

            data = [request.json]
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999

    @requires_helper_intent_model_type
    @requires_helper_intent_model
    @requires_json
    def delete(self):
        """
        刪除 intent_model
        Warning:會將robot相關的model資料都刪除(包含model. sentence. train test log) ！！！
        {
            "robot_id":"test.00001318",
            "modify_user": "charles.huang@quantatw.com"
        }
        """

        robot_id = request.json['robot_id']
        modify_user = request.json['modify_user']

        try:
            df = HelperIntentModelUtility.query_model(robot_id=robot_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.RobotNotExist,
                               message="This robot_id:{} has not existed.".format(robot_id),
                               data=[],
                               )

            HelperIntentModelUtility.delete_model(robot_id=robot_id,
                                                  modify_user=modify_user)

            # 相關log刪除
            HelperIntentModelUtility.delete_train_log(robot_id=robot_id,
                                                      modify_user=modify_user)
            HelperIntentModelUtility.delete_test_log(robot_id=robot_id,
                                                     modify_user=modify_user)
            HelperIntentModelUtility.delete_train_sentence_log(robot_id=robot_id,
                                                               modify_user=modify_user)
            HelperIntentModelUtility.delete_train_sentence(robot_id=robot_id,
                                                           modify_user=modify_user)
            HelperIntentModelUtility.delete_train_sentence_set(robot_id=robot_id,
                                                               modify_user=modify_user)

            # 實體model刪除
            path_offline = Get_MyEnv().env_fs_model_path + "intent/"
            img_confusion_matrix_path = Get_MyEnv().env_fs_image_path + "intent/confusion_matrix/"
            img_roc_path = Get_MyEnv().env_fs_image_path + "intent/roc/"
            HelperIntentModelUtility.remove_model_file_all_startwith(robot_id=robot_id,
                                                                     model_path_list=[path_offline,
                                                                                      img_confusion_matrix_path,
                                                                                      img_roc_path],
                                                                     modify_user=modify_user)

            data = [request.json]
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message='exception : ' + msg
                           ), 999


class IntentModelOnlineView(MethodView):

    @requires_helper_intent_model_online_type
    @requires_helper_intent_model_online
    @requires_json
    def post(self):
        """
        上線模型
        {
            "robot_id":"pepper.00001318",
            "model_id":"pepper.00001318_01",
            "modify_user": "charles.huang@quantatw.com",
            "online": true
        }
        """

        robot_id = request.json['robot_id']
        model_id = request.json['model_id']
        modify_user = request.json['modify_user']

        # online = true 上線 and false = 下線
        online = request.json['online']

        try:

            df = HelperIntentModelUtility.query_model(robot_id=robot_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.RobotNotExist,
                               message="This robot_id:{} has not existed.".format(robot_id),
                               data=[],
                               )

            df = HelperIntentModelUtility.query_train_log(robot_id=robot_id,
                                                          model_id=model_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.ModelNotExist,
                               message="This robot_id={} + model_id={} has not existed.".format(robot_id, model_id),
                               data=[],
                               )

            """
            實體model移動
            """
            path_online = Get_MyEnv().env_fs_model_path + "intent/online/"

            algorithm_type = df['algorithm_type'][0]
            if type(algorithm_type) == unicode:
                algorithm_type = algorithm_type.encode('utf8')

            move_file_success = True
            if online:
                status = 'Y'

                try:

                    # 移動到online資料夾, 舊model要先移除
                    # 從online資料夾移除
                    HelperIntentModelUtility.remove_model_file_all_startwith(robot_id=robot_id,
                                                                             model_path_list=[path_online])
                    # HelperIntentModelUtility.offline_model_file(robot_id, algorithm_type)
                    HelperIntentModelUtility.online_model_file(robot_id, model_id, algorithm_type)

                except IOError as ioe:
                    return jsonify(code=StateCode_HelperData.ModelNotExist,
                                   message="This robot_id:{robot_id} & model:{model_id} is fail to move to online file. Please check nfs intent_model file".format(
                                       robot_id=robot_id, model_id=model_id),
                                   data=[]
                                   )

                # robot有新的model上線
                HelperIntentModelUtility.update_model_by_column(robot_id=robot_id,
                                                                new_model_id=robot_id + "+" + model_id)

            else:
                status = 'N'

                try:
                    # 從online資料夾移除
                    HelperIntentModelUtility.remove_model_file_all_startwith(robot_id=robot_id,
                                                                             model_path_list=[path_online])
                    # HelperIntentModelUtility.offline_model_file(robot_id, algorithm_type)

                except IOError as ioe:
                    return jsonify(code=StateCode_HelperData.ModelNotExist,
                                   message="This robot_id:{robot_id} & model:{model_id} is fail to remove from online file. Please check nfs intent_model file".format(
                                       robot_id=robot_id, model_id=model_id),
                                   data=[]
                                   )

                # robot有新的model下線, 將new_model_id清空
                HelperIntentModelUtility.update_model_by_column(robot_id=robot_id,
                                                                new_model_id="")

            if move_file_success:
                # 改變訓練紀錄狀態
                HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                                    model_id=model_id,
                                                                    status=status,
                                                                    modify_user=modify_user)

            data = [request.json]
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999


class IntentAlgorithmView(MethodView):

    @requires_helper_intent_algorithm
    def get(self):
        """
        查詢 model
        api/helper/intentAlgorithm?algorithm_id=1dcnn
        """

        algorithm_id = request.args.get('algorithm_id', None)

        try:

            pd_df = HelperIntentModelUtility.query_algorithm(algorithm_id=algorithm_id)
            pd_df = pd_df[pd_df.in_use == True]

            create_user_list = list(pd_df.create_user.unique())
            # 重覆user過濾掉
            all_user_list = list(set(create_user_list))

            # 先將所有owner資料查出來
            user_info_dict = {}

            # 一次全撈
            info_list = HelperDataUtility.get_user(_camp_id=all_user_list)
            for info in info_list:
                user_info_dict.update({info['camp_user_id']: info})

            data = []
            for index, row in pd_df.iterrows():
                logger.debug(index)

                result = dict(
                    algorithm_id=row['algorithm_id'],
                    algorithm_name=row['algorithm_name'],
                    algorithm_type=row['algorithm_type'],
                    need_tokenizer=row['need_tokenizer'],
                    in_use=row['in_use'],
                    create_user=user_info_dict.get(row['create_user'], {}),
                    create_date=row['create_date']
                )
                data.append(result)

            # logger.debug('#####################')
            # logger.debug(data)
            # logger.debug(str(len(data)))

            # result = {
            #     "code": StateCode.Success,
            #     "message": "success",
            #     "data": data
            # }
            # return json.dumps(result, ensure_ascii=False)
            return jsonify(code=StateCode.Success,
                           message="success",
                           data=data
                           )

        except Exception as e:

            utility_logger = UtilityLogger()
            msg = utility_logger.except_error_msg(sys.exc_info())
            logger.error(msg)
            log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
            utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message=msg
                           ), 999


#
# class IntentModelPredictView(MethodView):
#
#     @requires_json
#     def post(self):
#         """
#         input
#         {
#             "input_text":["今天天氣好嗎？","現在國道北上塞車嗎"],
#             "robot_id":"hr.00001318"
#         }
#         """
#         input_text = request.json['input_text']
#         robot_id = request.json['robot_id']
#
#         """
#         安全檢查
#         """
#         df = HelperIntentModelUtility.query_model(robot_id=robot_id)
#         if len(df) == 0:
#             return jsonify(code=StateCode_HelperData.RobotNotExist,
#                            message="This robot_id:{} has not existed.".format(robot_id),
#                            data=[],
#                            )
#
#         """
#         model預測
#         """
#         df_all = pd.DataFrame({'sentence': input_text})
#
#         # 使用 predict class 預測
#         result = intent_predictor.predict(_robot_id=robot_id, _df_all=df_all)
#
#         # logger.debug(model_predict_df)
#
#         # 組成API回傳格式
#         data = []
#         for index, row in result['model_predict_df'].iterrows():
#             probability_list = row['y_predict_probability']
#             #     logger.debug(probability_list)
#             predict_list = range(0, len(result['mapping']))
#             #     logger.debug(predict_list)
#             predict_id = result['mapping']
#             #     logger.debug(predict_id)
#             predict_name = result['mapping_name']
#             # y_predict_id = row['y_predict_id']
#
#             probability_mapping_df = pd.DataFrame({'predict_index': predict_list,
#                                                    'predict_id': predict_id,
#                                                    'predict_name': predict_name,
#                                                    'confidence': probability_list})
#             #     logger.debug(probability_mapping_df)
#
#             # 依照信心值排序
#             probability_mapping_df = probability_mapping_df. \
#                 sort_values(by=['confidence'], ascending=False). \
#                 reset_index(drop=True)
#
#             # .T：表示以列的角度組資料; .values()：不要索引值
#             predict_result = probability_mapping_df.T.to_dict().values()
#
#             predict = dict(robot_id=robot_id,
#                            model_id=result['model_id'],
#                            mapping=result['mapping'],
#                            mapping_name=result['mapping_name'],
#                            sentence=row['sentence'],
#                            predict=predict_result)
#             data.append(predict)
#
#         return jsonify(code=StateCode.Success,
#                        message="success",
#                        data=data
#                        )
#
#         # # 組成API回傳格式
#         # data = []
#         # for index, row in model_predict_df.iterrows():
#         #     probability_list = row['y_predict_probability']
#         #     #     logger.debug(probability_list)
#         #     predict_list = range(0, len(mapping))
#         #     #     logger.debug(predict_list)
#         #     predict_id = mapping
#         #     #     logger.debug(predict_id)
#         #     predict_name = mapping_name
#         #
#         #     probability_mapping_df = pd.DataFrame({'predict_index': predict_list,
#         #                                            'predict_id': predict_id,
#         #                                            'predict_name': predict_name,
#         #                                            'confidence': probability_list})
#         #     #     logger.debug(probability_mapping_df)
#         #
#         #     # 依照信心值排序
#         #     probability_mapping_df = probability_mapping_df. \
#         #         sort_values(by=['confidence'], ascending=False). \
#         #         reset_index(drop=True)
#         #
#         #     # .T：表示以列的角度組資料; .values()：不要索引值
#         #     predict_result = probability_mapping_df.T.to_dict().values()
#         #
#         #     result = dict(robot_id=robot_id,
#         #                   sentence=row['sentence'],
#         #                   predict_by='model',
#         #                   predict=predict_result)
#         #
#         #     data.append(result)
#         #
#         # return jsonify(code=StateCode.Success,
#         #                message="success",
#         #                data=data
#         #                )
#
#         # try:
#         #
#         # except Exception as e:
#         #
#         #     # msg = 'exception : ' + str(e)
#         #     # msg = msg.replace("'", "")
#         #     # logger.error(msg)
#         #     # logger_db = UtilityLogger(CASSANDRA_ENV)
#         #     # logger_db.save_log(HELPER_KEYSPACE, HELPER_ERROR_LOG_TABLE, "GetDigitalFail", msg)
#         #
#         #     return jsonify(code=StateCode.Unexpected,
#         #                    data=[],
#         #                    message='exception : ' + str(e)
#         #                    ), 999


class IntentModelPredictTestView(MethodView):

    @requires_json
    def post(self):
        """
        input
        {
            "input_text":["今天天氣好嗎？","現在國道北上塞車嗎"],
            "robot_id":"pepper.00001318",
            "min_confidence":0.9,
            "model_id_list":["pepper.00001318_01"]
        }
        """

        # try:
        input_text = request.json['input_text']
        robot_id = request.json['robot_id']
        model_id_list = request.json['model_id_list']

        """
        安全檢查
        """
        df = HelperIntentModelUtility.query_model(robot_id=robot_id)
        if len(df) == 0:
            return jsonify(code=StateCode_HelperData.RobotNotExist,
                           message="This robot_id:{} has not existed.".format(robot_id),
                           data=[],
                           )

        for model_id in model_id_list:
            df = HelperIntentModelUtility.query_train_log(robot_id=robot_id,
                                                          model_id=model_id)
            if len(df) == 0:
                return jsonify(code=StateCode_HelperData.ModelNotExist,
                               message="This robot_id={} + model_id={} has not existed.".format(robot_id, model_id),
                               data=[],
                               )

        """
        model預測
        """
        df_all = pd.DataFrame({'sentence': input_text})

        # 預測
        # model預測測試用
        intent_test_predictor = Get_IntentModelPredictTest()
        # intent_test_predictor.load_model(_robot_id=robot_id,
        #                                  _model_id_list=model_id_list)
        result_list = intent_test_predictor.predict(_robot_id=robot_id,
                                                    _model_id_list=model_id_list,
                                                    _df_all=df_all)

        # logger.debug(model_predict_df)

        # 組成API回傳格式

        data_list = []
        for result in result_list:
            data = []
            for index, row in result['model_predict_df'].iterrows():
                probability_list = row['y_predict_probability']
                #     logger.debug(probability_list)
                predict_list = range(0, len(result['mapping']))
                #     logger.debug(predict_list)
                predict_id = result['mapping']
                #     logger.debug(predict_id)
                predict_name = result['mapping_name']
                # y_predict_id = row['y_predict_id']

                probability_mapping_df = pd.DataFrame({'predict_index': predict_list,
                                                       'predict_id': predict_id,
                                                       'predict_name': predict_name,
                                                       # 'y_predict_id': y_predict_id,
                                                       'confidence': probability_list})
                #     logger.debug(probability_mapping_df)

                # 依照信心值排序
                probability_mapping_df = probability_mapping_df. \
                    sort_values(by=['confidence'], ascending=False). \
                    reset_index(drop=True)

                # .T：表示以列的角度組資料; .values()：不要索引值
                predict_result = probability_mapping_df.T.to_dict().values()

                predict = dict(robot_id=robot_id,
                               model_id=result['model_id'],
                               mapping=result['mapping'],
                               mapping_name=result['mapping_name'],
                               sentence=row['sentence'],
                               predict=predict_result)
                data.append(predict)

            data_list.append(data)

        return jsonify(code=StateCode.Success,
                       message="success",
                       data=data_list
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


class IntentModelTrainingView(MethodView):

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

        # 一個機器人只能同時train一個model
        df = HelperIntentModelUtility.query_train_log(robot_id=robot_id)
        if len(df[df.status == 'TRAIN']) > 0:
            return jsonify(code=StateCode_HelperData.ModelInTraining,
                           message="This robot_id={} is training.".format(robot_id),
                           data=[],
                           )

        # model已經存在
        # df = HelperIntentModelUtility.query_train_log(robot_id=robot_id,
        #                                               model_id=model_id)
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
            #     print(upload_skill)
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

        # # clear_session() muti-thread training problem,
        # # 重複訓練會發生not an element of the graph問題, 必須清理session
        # from keras import backend as KS
        # KS.clear_session()
        graph = Graph()
        with graph.as_default():
            session = Session(graph=graph)
            with session.as_default():

                # 組出skill_name
                robot_skill_df = HelperIntentModelUtility.query_robot_skill(robot_id=robot_id)
                # 合併查詢
                sentence_df = pd.merge(sentence_df, robot_skill_df, how='inner', on=['skill_id'])
                sentence_df = sentence_df[['skill_id', 'skill_name', 'sentence_id', 'sentence', 'create_date_x']]

                # 只有一個skill, 使用語料庫句子, 組成兩個句子一起訓練
                one_skill = False
                other_skill_id = "_other"
                other_skill_name = "其他"
                # one_skill_id = "one_skill_id"

                if len(train_sentence_skill_sum) == 1:
                    logger.debug("======== one_skill =======")

                    # one_skill_id = sentence_df['skill_id'][0]

                    limit = len(sentence_df)
                    corpus_df = HelperIntentModelUtility.query_train_sentence_corpus(skill_id='gossip',
                                                                                     random=True,
                                                                                     limit=limit)
                    corpus_df['skill_id'] = other_skill_id
                    corpus_df['skill_name'] = other_skill_name
                    corpus_df['sentence_id'] = 'corpus'
                    corpus_df = corpus_df[['skill_id', 'skill_name', 'sentence_id', 'sentence']]
                    # logger.debug("=== corpus_df ===")
                    # logger.debug(corpus_df)
                    sentence_df = sentence_df.append(corpus_df)
                    one_skill = True

                # 演算法工廠
                if algorithm == "textcnn":
                    factory = IntentTextCnnModelFactory()
                    model_param = {
                        "embedding_output_dim": 256,
                        "vocab_size": 500,
                        "drop_out": 0.75,
                        "l2_reg_lambda": 0,
                        "batch_size": 15,
                        "epochs": 50,
                        "train_ratio": 0.9,
                        "early_stop": 30,
                        "optimizer": 'Adam',
                        "loss": 'sparse_categorical_crossentropy',
                        "model_sub_name": ".h5",
                        "tokenizer_sub_name": ".pickle",
                        "verbose": 1
                    }

                    # model_param = IntentTextCnnModelParam(sentence_max_len=sentence_max_len,
                    #                                       embedding_output_dim=256,
                    #                                       drop_out=0.75,
                    #                                       l2_reg_lambda=0.0,
                    #                                       batch_size=15,
                    #                                       epochs=50,
                    #                                       train_ratio=0.9,
                    #                                       early_stop=30,
                    #                                       optimizer='Adam',
                    #                                       loss='sparse_categorical_crossentropy',
                    #                                       )

                elif algorithm == "1dcnn":
                    factory = IntentOnedCnnModelFactory()
                    model_param = {
                        "filters": 512,
                        "embedding_output_dim": 256,
                        "vocab_size": 500,
                        "batch_size": 15,
                        "epochs": 50,
                        "train_ratio": 0.9,
                        "early_stop": 30,
                        "optimizer": 'Adam',
                        "loss": 'sparse_categorical_crossentropy',
                        "kernel_size": 3,
                        "model_sub_name": ".h5",
                        "tokenizer_sub_name": ".pickle",
                        "verbose": 1
                    }

                    # model_param = IntentOnedCnnModelParam(sentence_max_len=sentence_max_len,
                    #                                       embedding_output_dim=256,
                    #                                       kernel_size=3,
                    #                                       filters=512,
                    #                                       batch_size=15,
                    #                                       epochs=50,
                    #                                       train_ratio=0.9,
                    #                                       early_stop=30,
                    #                                       optimizer='Adam',
                    #                                       loss='sparse_categorical_crossentropy',
                    #                                       )

                elif algorithm == "2dcnn":
                    factory = IntentTwodCnnModelFactory()
                    model_param = {
                        "filter_sizes": [2, 3, 5],
                        "num_filters": 512,
                        "embedding_output_dim": 256,
                        "vocab_size": 500,
                        "drop_out": 0.5,
                        "batch_size": 15,
                        "epochs": 50,
                        "train_ratio": 0.9,
                        "early_stop": 30,
                        "optimizer": 'Adam',
                        "loss": 'sparse_categorical_crossentropy',
                        "model_sub_name": ".h5",
                        "tokenizer_sub_name": ".pickle",
                        "verbose": 1
                    }

                    # model_param = IntentTwodCnnModelParam(sentence_max_len=sentence_max_len,
                    #                                       embedding_output_dim=256,
                    #                                       drop_out=0.5,
                    #                                       filter_sizes=[2, 3, 5],
                    #                                       num_filters=512,
                    #                                       batch_size=15,
                    #                                       epochs=50,
                    #                                       train_ratio=0.9,
                    #                                       early_stop=30,
                    #                                       optimizer='Adam',
                    #                                       loss='sparse_categorical_crossentropy',
                    #                                       )

                elif algorithm == "gru":
                    factory = IntentGRUModelFactory()
                    model_param = {
                        "embedding_output_dim": 256,
                        "vocab_size": 500,
                        "drop_out": 0.5,
                        "batch_size": 15,
                        "epochs": 50,
                        "train_ratio": 0.9,
                        "early_stop": 30,
                        "optimizer": 'Adam',
                        "loss": 'sparse_categorical_crossentropy',
                        "model_sub_name": ".h5",
                        "tokenizer_sub_name": ".pickle",
                        "verbose": 1
                    }
                    # model_param = IntentGRUModelParam(sentence_max_len=sentence_max_len,
                    #                                   embedding_output_dim=256,
                    #                                   drop_out=0.2,
                    #                                   batch_size=15,
                    #                                   epochs=50,
                    #                                   train_ratio=0.9,
                    #                                   early_stop=30,
                    #                                   optimizer='Adam',
                    #                                   loss='sparse_categorical_crossentropy',
                    #                                   )

                elif algorithm == "rf":
                    # randomforest
                    factory = IntentRandomForestModelFactory()
                    model_param = {
                        "n_estimators": 20,
                        "max_depth": [3, None],
                        "criterion": ['gini'],
                        "model_sub_name": ".joblib",
                        "tokenizer_sub_name": ".pickle"
                    }
                    # model_param = IntentRandomForestModelParam()

                elif algorithm == "lg":
                    # logisticregression
                    factory = IntentLogisticRegressionModelFactory()
                    model_param = {
                        "c": [10],
                        "penalty": ['l2'],
                        "model_sub_name": ".joblib",
                        "tokenizer_sub_name": ".pickle"
                    }

                    # model_param = IntentLogisticRegressionModelParam()

                # elif algorithm == "bert":
                #     # bert
                #     factory = IntenBertModelFactory()
                #     model_param = {
                #         "bert_base": "bert-base-chinese",
                #         "batch_size": 16,
                #         "epochs": 3,
                #         "sentence_max_len": 20,
                #         "no_decay": ['bias', 'gamma', 'beta'],
                #         "attention_masks_column": 'attention_masks',
                #         "optimizer_name": 'BertAdam',
                #         "model_sub_name": ".bin",
                #         "tokenizer_sub_name": ".pickle"
                #     }
                #
                #     # model_param = IntentBertModelParam(num_classes=0,
                #     #                                    bert_base="bert-base-chinese",
                #     #                                    batch_size=16,
                #     #                                    epochs=3,
                #     #                                    attention_masks_column='attention_masks',
                #     #                                    no_decay=['bias', 'gamma', 'beta'])

                else:
                    error_msg = "algorithm : {algorithm} has not implemented !!".format(algorithm=algorithm)
                    logger.error(error_msg)
                    data = [request.json]
                    return jsonify(code=StateCode_HelperData.ModelNotExist,
                                   message=error_msg,
                                   data=data
                                   )

                # 每個演算法都有的參數
                model_param.update(
                    {
                        "num_classes": 0,
                        "sentence_max_len": sentence_max_len,
                        "train_test_split_size": train_test_split_size
                    })

                """
                建立模型,使用create_model(), 將model new出來
                EX：LogisticRegression就有好幾種變化的model, 未來可以入參數生產不同的LogisticRegression model
                """
                logger.debug('create_model()')
                model = factory.create_model()

                sentence_column = 'sentence'
                preprocessing_column = 'cut_words'
                feature_column = 'feature'
                target_column = 'skill_id'
                target_index_column = 'target_index'

                logger.debug('preprocessing')

                # 資料前處理
                sentence_df = model.preprocessing(sentence_df=sentence_df,
                                                  input_column=sentence_column,
                                                  output_column=preprocessing_column)

                logger.debug('mapping_setting')

                # target(EX: weather)必須轉成index(EX:0), model才能使用, 保留對應的mapping將來轉回target文字(0 -> weather)
                sentence_df = model.mapping_setting(sentence_df=sentence_df,
                                                    input_column=target_column,
                                                    output_column=target_index_column)

                logger.debug('feature_engineering')

                # 特徵工程
                feature_transformer_path = Get_MyEnv().env_fs_model_path + "intent/"
                feature_transformer_name = robot_id + "+" + model_id
                sentence_df = model.feature_engineering(sentence_df=sentence_df,
                                                        input_column=preprocessing_column,
                                                        output_column=feature_column,
                                                        model_param=model_param)

                logger.debug('save_transformer')

                # 訓練時將字典存到fs, DL預測時必須使用
                model.save_transformer(feature_transformer_path, feature_transformer_name)

                logger.debug('data_split')

                # target(EX: weather)必須轉成index(EX:0), model才能使用, 保留對應的mapping將來轉回target文字(0 -> weather)
                x_train, y_train, _, test_df = model.data_split(sentence_df=sentence_df,
                                                                feature_column=feature_column,
                                                                target_index_column=target_index_column,
                                                                train_test_split_size=train_test_split_size,
                                                                one_skill=one_skill,
                                                                other_skill_id=other_skill_id,
                                                                model_param=model_param)

                # logger.debug('sentence_df')
                # logger.debug(sentence_df)

                # logger.debug(model.mapping)
                # logger.debug(model.mapping_name)

                num_classes = model.get_num_classes()
                # logger.debug(num_classes)

                # 前處理後才知道的參數
                model_param['num_classes'] = num_classes
                # logger.debug(model_param.num_classes)

                """
                build model : model參數與model合併
                """
                logger.debug('============ build_model ============')
                model.build_model(model_param)

                # 訓練參數轉成json保存(object >> dict >> json)
                # algorithm_param_json = json.dumps(model.model_param.__dict__) #class轉dict
                algorithm_param_json = json.dumps(model.model_param)

                # 訓練模型時, 新增訓練紀錄, 狀態爲 TRAIN 訓練中

                logger.debug('============ insert_train_log ============')

                nfs_model_id = robot_id + "+" + model_id + model_param['model_sub_name']
                nfs_tokenizer_id = robot_id + "+" + model_id + model_param['tokenizer_sub_name']

                HelperIntentModelUtility.insert_train_log(robot_id=robot_id,
                                                          model_id=model_id,
                                                          sentence_set_id=sentence_set_id,
                                                          algorithm=algorithm,
                                                          modify_user=modify_user,
                                                          algorithm_param=algorithm_param_json,
                                                          algorithm_type=model.algorithm_type,
                                                          tokenizer_id=feature_transformer_name,
                                                          mapping=model.mapping,
                                                          mapping_name=model.mapping_name,
                                                          train_test_size=train_test_split_size,
                                                          status="TRAIN",
                                                          nfs_model_id=nfs_model_id,
                                                          nfs_tokenizer_id=nfs_tokenizer_id
                                                          )

                # TODO: TRAIN 開始就回前端 , 非同步去呼叫train model API 不要等
                try:

                    logger.debug('============ train_model_start ============')

                    """
                    訓練模型
                    """
                    model.train_model(x_train,
                                      y_train)

                    # graph = Graph()
                    # with graph.as_default():
                    #     session = Session(graph=graph)
                    #     with session.as_default():
                    #         model.train_model(x_train=x_train,
                    #                           y_train=y_train)
                    #
                    #         logger.debug('============ save_session ============')
                    #         robot_id_model_id = robot_id + "+" + model_id
                    #         intent_test_predictor.model_dict.update({robot_id_model_id: model})
                    #         intent_test_predictor.graph_dict.update({robot_id_model_id: graph})
                    #         intent_test_predictor.session_dict.update({robot_id_model_id: session})

                    # robot_id_model_id = robot_id + "+" + model_id
                    # session_data = intent_test_predictor.session_dict.get(robot_id_model_id, None)
                    # if session_data:
                    #     KS.set_session(intent_test_predictor.session_dict[robot_id_model_id])
                    #     with intent_test_predictor.graph_dict[robot_id_model_id].as_default():
                    #         model.train_model(x_train=x_train,
                    #                           y_train=y_train)
                    # else:
                    #     model.train_model(x_train=x_train,
                    #                       y_train=y_train)
                    #
                    #     graph = Graph()
                    #     with graph.as_default():
                    #         session = Session(graph=graph)
                    #         with session.as_default():
                    #
                    #             logger.debug('============ save_session ============')
                    #             robot_id_model_id = robot_id + "+" + model_id
                    #             intent_test_predictor.model_dict.update({robot_id_model_id: model})
                    #             intent_test_predictor.graph_dict.update({robot_id_model_id: graph})
                    #             intent_test_predictor.session_dict.update({robot_id_model_id: session})
                    #
                    # logger.debug('============ train_model_finish ============')

                    """
                    model evaluate
                    直接把訓練句丟model預測, 切出驗證資料會使訓練資料過少(TODO: 如果能有夠多句子或自動生成句子就可以切)
                    並把結果計算統計值 EX : accuracy, f1-score 用來參考
                    """
                    predict_df = model.predict_result(test_df=test_df,
                                                      sentence_column=sentence_column,
                                                      feature_column=feature_column,
                                                      target_column=target_column,
                                                      target_index_column=target_index_column,
                                                      model_param=model_param)

                    y_predict_name = predict_df['y_predict_name']

                    logger.debug("----- 統計值 -----")
                    logger.debug(predict_df)

                    answer = predict_df['answer']
                    predict = predict_df['y_predict']

                    # 預測錯誤
                    logger.debug("----- 分類錯誤的類別 -----")

                    if one_skill:
                        # 單一技能設定門檻值當成錯誤分類
                        # error_df_1 = predict_df[predict_df['y_predict'] != one_skill_id]
                        error_df_1 = predict_df[predict_df['y_predict'] != predict_df['answer']]
                        error_df_2 = predict_df[predict_df['y_predict'] == predict_df['answer']]

                        default_one_skill_confidence = 0.9
                        greate_than_default_confidece = []
                        for index, row in error_df_2.iterrows():
                            if max(row["y_predict_probability"]) >= default_one_skill_confidence:
                                greate_than_default_confidece.append(True)
                            else:
                                greate_than_default_confidece.append(False)

                        error_df_2['greate_than_default_confidece'] = greate_than_default_confidece
                        # 信心值過低都算錯
                        error_df_2 = error_df_2[error_df_2['greate_than_default_confidece'] == False]
                        error_df = error_df_1.append(error_df_2)

                    else:
                        # 多類別
                        error_df = predict_df[predict_df['y_predict'] != predict_df['answer']]

                    # logger.debug(error_df)

                    total_count = len(test_df)
                    error_count = len(error_df)
                    correct_count = total_count - error_count
                    # scikit-learn提供的accuracy_score, 會在分母乘上與權重
                    # acc_score = accuracy_score(answer, predict)
                    # 自己計算的accuracy_score
                    acc_score = float(correct_count) / total_count
                    # acc_score = float(
                    #     sum(np.array(sentence_df.skill_id) == np.argmax(y_predict_probability, axis=1))) / len(
                    #     sentence_df)
                    logger.debug('total_count: {}'.format(total_count))
                    logger.debug('correct_count: {}'.format(correct_count))

                    if one_skill:
                        # 無數值
                        precision_score, recall_score, f1_score = -1, -1, -1

                    else:

                        # confusion matrix
                        logger.debug("----- confusion matrix -----")
                        logger.debug(pd.crosstab(np.array(y_predict_name), np.array(test_df.skill_id),
                                                 rownames=['predict'], colnames=['answer']))

                        """
                        accuracy.precision.recall.f1_score
                        """
                        precision_score, recall_score, f1_score, support = precision_recall_fscore_support(answer,
                                                                                                           predict,
                                                                                                           average='macro')

                    acc_score = float('%.4f' % acc_score)
                    precision_score = float('%.4f' % precision_score)
                    recall_score = float('%.4f' % recall_score)
                    f1_score = float('%.4f' % f1_score)

                    logger.debug('accuracy: {0:0.4f}'.format(acc_score))
                    logger.debug('precision: {0:0.4f}'.format(precision_score))
                    logger.debug('recall: {0:0.4f}'.format(recall_score))
                    logger.debug('f1_score: {0:0.4f}'.format(f1_score))

                except TypeError as e:

                    # 訓練模型失敗, 將train_log刪除
                    HelperIntentModelUtility.delete_train_log(robot_id=robot_id,
                                                              model_id_list=[model_id],
                                                              modify_user=modify_user)

                    utility_logger = UtilityLogger()
                    msg = utility_logger.except_error_msg(sys.exc_info())
                    logger.error(msg)
                    log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
                    utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

                    return jsonify(code=StateCode.Unexpected,
                                   data=[],
                                   message=msg
                                   ), 999

                try:
                    """
                    持久化實體檔案
                    1. 模型
                    2. 評估 - ROC圖
                    3. 評估 - confusion_matrix圖
                    4. 評估 - PRC圖
                    5. 模型 - 訓練過程
                    """
                    save_model_path = Get_MyEnv().env_fs_model_path + "intent/"
                    model_name = robot_id + "+" + model_id
                    model.save_model(save_model_path, model_name)

                    plt_name = robot_id + "+" + model_id + ".png"

                    train_history_plt = model.train_history_plt(train='loss',
                                                                validation='val_loss')
                    if train_history_plt:
                        train_history_path = Get_MyEnv().env_fs_image_path + "intent/train_history/" + plt_name
                        train_history_plt_url = Get_MyEnv().env_fs_image_url + "intent/train_history/" + plt_name
                        train_history_plt.savefig(train_history_path,
                                                  bbox_inches='tight')
                    else:
                        train_history_plt_url = ''

                    if one_skill == False:

                        # logger.debug(predict_df)

                        cm_plt = skplt.metrics.plot_confusion_matrix(list(predict_df['answer_id']),
                                                                     list(predict_df['y_predict_id']))

                        cm_plt_path = Get_MyEnv().env_fs_image_path + "intent/confusion_matrix/" + plt_name
                        cm_plt_url = Get_MyEnv().env_fs_image_url + "intent/confusion_matrix/" + plt_name
                        cm_plt.get_figure().savefig(cm_plt_path,
                                                    bbox_inches='tight')

                        # logger.debug(list(predict_df['answer_id']))
                        # logger.debug(list(predict_df['y_predict_probability']))

                        roc_plt = skplt.metrics.plot_roc(list(predict_df['answer_id']),
                                                         list(predict_df['y_predict_probability']))

                        roc_plt_path = Get_MyEnv().env_fs_image_path + "intent/roc/" + plt_name
                        roc_plt_url = Get_MyEnv().env_fs_image_url + "intent/roc/" + plt_name

                        roc_plt.get_figure().savefig(roc_plt_path,
                                                     bbox_inches='tight')

                        prc_plt = skplt.metrics.plot_precision_recall_curve(list(predict_df['answer_id']),
                                                                            list(predict_df[
                                                                                     'y_predict_probability']),
                                                                            cmap='nipy_spectral')

                        prc_plt_path = Get_MyEnv().env_fs_image_path + "intent/prc/" + plt_name
                        prc_plt_url = Get_MyEnv().env_fs_image_url + "intent/prc/" + plt_name
                        prc_plt.get_figure().savefig(prc_plt_path,
                                                     bbox_inches='tight')

                        prc_plt.cla()
                        pie_plt_url = ''


                    else:
                        cm_plt_url = ''
                        roc_plt_url = ''
                        prc_plt_url = ''
                        labels = ['Correct', 'Others']
                        sizes = [correct_count, error_count]
                        colors = ['#66b3ff', '#ff9999']
                        explode = (0, 0.1)
                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes,
                                explode=explode, labels=labels, autopct='%1.1f%%',
                                shadow=False, startangle=90, colors=colors)
                        ax1.axis('equal')
                        # plt.tight_layout()

                        pie_plt_path = Get_MyEnv().env_fs_image_path + "intent/pie/" + plt_name
                        pie_plt_url = Get_MyEnv().env_fs_image_url + "intent/pie/" + plt_name
                        plt.savefig(pie_plt_path,
                                    bbox_inches='tight')
                        plt.cla()

                    plt_img_url_dict = {}
                    if cm_plt_url != '':
                        plt_img_url_dict.update({"Confusion Matrix": cm_plt_url})
                    if train_history_plt_url != '':
                        plt_img_url_dict.update({"TRAIN HISTORY": train_history_plt_url})
                    if roc_plt_url != '':
                        plt_img_url_dict.update({"ROC": roc_plt_url})
                    if prc_plt_url != '':
                        plt_img_url_dict.update({"PRC": prc_plt_url})
                    if pie_plt_url != '':
                        plt_img_url_dict.update({"PIE": pie_plt_url})

                    # plt_img_url_dict = {
                    #     "Confusion Matrix": cm_plt_url,
                    #     "TRAIN HISTORY": train_history_plt_url,
                    #     "ROC": roc_plt_url,
                    #     "PRC": prc_plt_url,
                    #     "PIE": pie_plt_url
                    # }
                    plt_img_url_json = json.dumps(plt_img_url_dict, ensure_ascii=False)

                except IOError as e:

                    # 訓練模型失敗, 將train_log刪除
                    HelperIntentModelUtility.delete_train_log(robot_id=robot_id,
                                                              model_id_list=[model_id],
                                                              modify_user=modify_user)

                    utility_logger = UtilityLogger()
                    msg = utility_logger.except_error_msg(sys.exc_info())
                    logger.error(msg)
                    log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
                    utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

                    return jsonify(code=StateCode.Unexpected,
                                   data=[],
                                   message=msg
                                   ), 999

                # TODO: 非同步 following task

                # 將model狀態 status=TRAIN to N
                HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                                    model_id=model_id,
                                                                    status="N"
                                                                    )

                # 回傳驗證錯誤句子
                error_evaluate_list = []
                error_evaluate_sentence = []
                for index, row in error_df.iterrows():

                    # 其他類驗證錯誤不顯示
                    if one_skill and row.answer_id == other_skill_id:
                        continue

                    error = dict(
                        sentence=row.sentence,
                        predict_id=row.y_predict_id,
                        answer_id=row.answer_id,
                    )
                    error_evaluate_list.append(error)
                    error_evaluate_sentence.append(row.sentence)

                # 保存驗證錯誤句子
                sentence = list(error_df.sentence)
                predict_id = list(error_df.y_predict_id)
                answer_id = list(error_df.answer_id)
                save_error_evaluate = dict(
                    sentence=sentence,
                    predict_id=predict_id,
                    answer_id=answer_id,
                )
                # 轉成json保存
                save_error_evaluate_json = json.dumps(save_error_evaluate)
                # logger.debug(save_error_evaluate_json)

                # logger.debug('@@@@@@ 3 insert_test_log over')

                HelperIntentModelUtility.insert_test_log(robot_id=robot_id,
                                                         model_id=model_id,
                                                         total_count=total_count,
                                                         correct_count=correct_count,
                                                         accuracy_score=acc_score,
                                                         precision_score=precision_score,
                                                         recall_score=recall_score,
                                                         f1_score=f1_score,
                                                         img=plt_img_url_json,
                                                         modify_user=modify_user,
                                                         error_evaluate_sentence=save_error_evaluate_json
                                                         )

                # logger.debug('@@@@@@ 3 insert_test_log over')

                # 此次訓練語句紀錄
                if one_skill:
                    sentence_df = sentence_df[sentence_df['skill_id'] != other_skill_id]

                cql_list = []
                param_tuple_list = []
                for index, row in sentence_df.iterrows():
                    cql, param_tuple = HelperIntentModelUtility.insert_train_sentence_log_get_cql(robot_id=robot_id,
                                                                                                  model_id=model_id,
                                                                                                  skill_id=row[
                                                                                                      'skill_id'],
                                                                                                  sentence_id=row[
                                                                                                      'sentence_id'],
                                                                                                  sentence=row[
                                                                                                      'sentence'],
                                                                                                  cut_sentence=row[
                                                                                                      'cut_words'],
                                                                                                  create_date=row[
                                                                                                      'create_date_x']
                                                                                                  )
                    cql_list.append(cql)
                    param_tuple_list.append(param_tuple)

                    # HelperIntentModelUtility.insert_train_sentence_log(robot_id=robot_id,
                    #                                                    model_id=model_id,
                    #                                                    skill_id=row['skill_id'],
                    #                                                    sentence_id=row['sentence_id'],
                    #                                                    sentence=row['sentence'],
                    #                                                    cut_sentence=row['cut_words'],
                    #                                                    create_date=row['create_date_x']
                    #                                                    )

                if len(cql_list) > 0:
                    # HelperIntentModelUtility.exec_cql_transations(cql_list)
                    HelperIntentModelUtility.exec_cql_transations_param_tuple(cql_list, param_tuple_list)

                data = dict(
                    accuracy_score=acc_score,
                    precision_score=recall_score,
                    recall_score=recall_score,
                    f1_score=f1_score,
                    error_evaluate_sentence=error_evaluate_list
                )

                return jsonify(code=StateCode.Success,
                               message="success",
                               data=[data]
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


# GraphQL 測試
class UserField(ObjectType):
    campUserId = String()
    companyId = String()
    employeeChtName = String()
    employeeEngName = String()
    employeeId = String()
    extNumber = String()


class IntentModelField(ObjectType):
    robotId = String()
    createUser = Field(UserField)
    createDate = String()
    modifyUser = Field(UserField)
    modifyDate = String()
    minConfidence = String()


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


class IntentModelQuery(ObjectType):
    # query string
    models = List(IntentModelField,
                  robotId=String(required=True),
                  modelId=String(required=False))

    def resolve_models(self, info, robotId=None):
        logger.debug('resolve_models')
        return get_model(robot_id=robotId)


if __name__ == "__main__":

    # log
    # 系統log只顯示error級別以上的
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    # 自訂log
    logger = logging.getLogger('FlaskAPi_HelperIntentModel.py')
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

    # init jieba
    jieba = Get_JiebaSegmentor()

    # dao = MssqlDAO(ip=Get_MyEnv().env_mssql_ip,
    #                account=Get_MyEnv().env_mssql_account,
    #                password=Get_MyEnv().env_mssql_password)

    # dao = envCassandraDAO(Get_MyEnv().env_cassandra)
    # p_dao = CassandraDAO(CassandraType.PRODUCTION)
    # t_dao = CassandraDAO(CassandraType.TEST)
    # m_dao = MssqlDAO()

    # # # model預測用
    # intent_predictor = Get_IntentModelPredict()

    # # model預測測試用
    intent_test_predictor = Get_IntentModelPredictTest()

    app.add_url_rule('/api/helper/userRobot',
                     view_func=UserRobotView.as_view('UserRobotView'))

    app.add_url_rule('/api/helper/intentModel',
                     view_func=IntentModelView.as_view('IntentModelView'))

    app.add_url_rule('/api/helper/intentModelOnline',
                     view_func=IntentModelOnlineView.as_view('IntentModelOnlineView'))

    app.add_url_rule('/api/helper/intentSentence',
                     view_func=IntentSentenceView.as_view('IntentSentenceView'))

    app.add_url_rule('/api/helper/intentTrainSentenceSet',
                     view_func=IntentTrainSentenceSetView.as_view('IntentTrainSentenceSetView'))

    app.add_url_rule('/api/helper/intentTrainSentenceSetCopy',
                     view_func=IntentTrainSentenceSetCopyView.as_view('IntentTrainSentenceSetCopyView'))

    app.add_url_rule('/api/helper/intentTrainSentence',
                     view_func=IntentTrainSentenceView.as_view('IntentTrainSentenceView'))

    app.add_url_rule('/api/helper/intentTrainSentenceWithFile',
                     view_func=IntentTrainSentenceWithFileView.as_view('IntentTrainSentenceWithFileView'))

    app.add_url_rule('/api/helper/intentTrainSentenceExportFile',
                     view_func=IntentTrainSentenceExportFileView.as_view('IntentTrainSentenceExportFileView'))

    app.add_url_rule('/api/helper/intentTrainSentenceLog',
                     view_func=IntentTrainSentenceLogView.as_view('IntentTrainSentenceLogView'))

    app.add_url_rule('/api/helper/sentenceModifyLog',
                     view_func=SentenceModifyLogView.as_view('SentenceModifyLogView'))

    app.add_url_rule('/api/helper/semanticModelLog',
                     view_func=SemanticModelLogView.as_view('SemanticModelLogView'))

    app.add_url_rule('/api/helper/intentTrainLog',
                     view_func=IntentTrainLogView.as_view('IntentTrainLogView'))

    app.add_url_rule('/api/helper/intentTestLog',
                     view_func=IntentTestLogView.as_view('IntentTestLogView'))

    app.add_url_rule('/api/helper/intentTrainTestLog',
                     view_func=IntentTrainTestLogView.as_view('IntentTrainTestLogView'))

    app.add_url_rule('/api/helper/intentAlgorithm',
                     view_func=IntentAlgorithmView.as_view('IntentAlgorithmView'))

    # app.add_url_rule('/api/helper/intentModelPredict',
    #                  view_func=IntentModelPredictView.as_view('IntentModelPredictView'))

    app.add_url_rule('/api/helper/intentModelPredictTest',
                     view_func=IntentModelPredictTestView.as_view('IntentModelPredictTestView'))

    app.add_url_rule('/api/helper/intentModelTraining',
                     view_func=IntentModelTrainingView.as_view('IntentModelTrainingView'))
    # app.add_url_rule('/api/helper/IntentModelEvaluate',
    #                  view_func=IntentModelEvaluateView.as_view('IntentModelEvaluateView'))

    # GraphQL 測試
    schema = Schema(query=IntentModelQuery)
    app.add_url_rule('/api/helper/intentModelGql',
                     view_func=GraphQLView.as_view(
                         'IntentModelGqlView',
                         schema=schema,
                         graphiql=True))

    logger.info('api run !!')

    if Get_MyEnv().env_oo == Get_MyEnv().LOCAL:
        app.run(port=5011, host='192.168.0.41', debug=False, use_reloader=False, threaded=True)
    else:
        app.run(port=5011, host='0.0.0.0', debug=True, use_reloader=False, threaded=True)

    # HelperData 5005
    # SparkSubmit_HelperData 5006
    # HelperDataModel 5007
    # HelperData_mssql 5010
    # HelperDataModel_mssql 5011
