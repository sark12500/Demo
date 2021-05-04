# encoding=UTF-8
# !flask/bin/python

import logging
from datetime import datetime
import glob, os, shutil, json
from threading import Lock, Thread
# import torch
# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from keras import backend as KS
from tensorflow import Graph, Session

from Config_Helper import Get_HelperConfig
from HelperIntentModelUtility import HelperIntentModelUtility
from NLP_IntentModelFactory import *

def singleton(clz):
    instances = {}

    def getinstance(*args, **kwargs):
        if clz not in instances:
            instances[clz] = clz(*args, **kwargs)

        return instances[clz]

    return getinstance


def Get_IntentModelPredictTest():
    return IntentModelPredictTest()


# @singleton
class IntentModelPredictTest(object):

    def __init__(self):

        # log
        # 系統log只顯示error級別以上的
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # 自訂log
        self.logger = logging.getLogger('NLP_IntentModelPredictir.py')
        self.logger.setLevel(logging.DEBUG)

        config = Get_HelperConfig()
        self.HELPER_KEYSPACE = Get_MyEnv().env_helper_keyspace
        self.HELPER_INTENT_MODEL_TABLE = config.HELPER_INTENT_MODEL_TABLE
        self.HELPER_INTENT_TRAIN_LOG_TABLE = config.HELPER_INTENT_TRAIN_LOG_TABLE

        # 存放共用model
        self.model_dict = {}
        self.tokenizer_dict = {}
        self.graph_dict = {}
        self.session_dict = {}

        self.js = Get_JiebaSegmentor()
        # self.c_dao = envCassandraDAO(Get_MyEnv().env_cassandra)

        # mutithread下更新共用變數Lock住
        self.lock = Lock()

    def get_mapping_data(self):
        return self.mapping

    def load_model(self, _robot_id, _model_id_list):
        """
        讀取folder存在的model
        ISSUE：無法在init直接下cassandra query(insert, update可以) ~ WTF???
        目前解決方法：
        1. 選擇讀取model資料夾(model/intent/online/...)
        2. online放了最新上線的model , load_model沒指定robot_id讀取全部, 有的話讀取該robot_id的model
        3  將 new_model_id 改回 robot_id+model_if
        :param _robot_id:
        :param _model_id_list:
        :return:
        """

        # from keras import backend as KS
        # KS.clear_session()

        print("============load_model=============")
        if _robot_id:
            print("_robot_id : {}".format(_robot_id))

        api_start = datetime.now()

        # model對應位置
        path = Get_MyEnv().env_fs_model_path + "intent/"

        # robot_id開頭的檔案讀出來
        load_dl_model_name = _robot_id + "+*.h5"
        load_ml_model_name = _robot_id + "+*.joblib"
        load_bert_model_name = _robot_id + "+*.bin"
        load_tokenizer_name = _robot_id + "+*.pickle"

        self.logger.debug("load_dl_model_name : {} ".format(os.path.join(path, load_dl_model_name)))
        self.logger.debug("load_ml_model_name : {} ".format(os.path.join(path, load_ml_model_name)))
        self.logger.debug("load_bert_model_name : {} ".format(os.path.join(path, load_bert_model_name)))
        self.logger.debug("load_tokenizer_path : {} ".format(os.path.join(path, load_tokenizer_name)))

        fs_model_list = glob.glob(os.path.join(path, load_dl_model_name)) + \
                        glob.glob(os.path.join(path, load_ml_model_name)) + \
                        glob.glob(os.path.join(path, load_bert_model_name))

        for model_name in fs_model_list:
            for m_id in _model_id_list:

                if model_name == os.path.join(path, _robot_id + "+" + m_id + ".h5") or \
                        model_name == os.path.join(path, _robot_id + "+" + m_id + ".joblib") or \
                        model_name == os.path.join(path, _robot_id + "+" + m_id + ".bin"):

                    """
                    恢覆模型的線程和預測時的線程不一致，導致graph不一樣?
                    >> 在恢覆模型時保存此時的graph, 不同model要有不同graph
                    """
                    graph = Graph()
                    with graph.as_default():
                        session = Session(graph=graph)
                        with session.as_default():
                            if os.path.splitext(model_name)[-1] == ".h5":
                                model = load_model(model_name)
                            elif os.path.splitext(model_name)[-1] == ".joblib":
                                model = load(model_name)
                            # elif os.path.splitext(model_name)[-1] == ".bin":
                            #     model_state_dict = torch.load(model_name)
                            #
                            #     train_log_pd = HelperIntentModelUtility.query_train_log(robot_id=_robot_id,
                            #                                                             model_id=m_id)
                            #
                            #     # self.logger.debug(train_log_pd)
                            #
                            #     if len(train_log_pd) > 0:
                            #         algorithm_param = train_log_pd['algorithm_param'][0]
                            #
                            #         algorithm_param_dict = json.loads(algorithm_param)
                            #         bert_base = algorithm_param_dict['bert_base']
                            #         num_classes = algorithm_param_dict['num_classes']
                            #
                            #         model = BertForSequenceClassification.from_pretrained(bert_base,
                            #                                                               state_dict=model_state_dict,
                            #                                                               num_labels=num_classes)
                            #     else:
                            #         self.logger.debug('{model_name} has no train log'.format(model_name=model_name))

                            else:
                                self.logger.debug('no model !!!!')
                                continue

                            self.logger.debug('os.path.basename(model_name) : {}'.format(os.path.basename(model_name)))
                            with self.lock:
                                self.logger.debug('-------- Lock - update memory model --------')
                                self.model_dict.update({os.path.basename(model_name): model})
                                self.graph_dict.update({os.path.basename(model_name): graph})
                                self.session_dict.update({os.path.basename(model_name): session})
                                self.logger.debug('-------- unLock --------')

                        break

                # glob無法透過檔名直接過濾, 所以查出全部model再過濾
                # self.logger.debug('model_name')
                # self.logger.debug(model_name)
                # self.logger.debug('os.path.join(path, _robot_id + "+" + m_id + ".h5")')
                # self.logger.debug(os.path.join(path, _robot_id + "+" + m_id + ".h5"))

                # if model_name == os.path.join(path, _robot_id + "+" + m_id + ".h5") or \
                #         model_name == os.path.join(path, _robot_id + "+" + m_id + ".joblib"):
                #
                #
                #     if os.path.splitext(model_name)[-1] == ".h5":
                #         model = load_model(model_name)
                #     elif os.path.splitext(model_name)[-1] == ".joblib":
                #         model = load(model_name)
                #     else:
                #         self.logger.debug('no model !!!!')
                #         continue
                #
                #     self.logger.debug('os.path.basename(model_name) : {}'.format(os.path.basename(model_name)))
                #     self.model_dict.update({os.path.basename(model_name): model})
                #     break

        self.logger.debug("model_dict : ")
        self.logger.debug(self.model_dict)

        for tokenizer_name in glob.glob(os.path.join(path, load_tokenizer_name)):
            for m_id in _model_id_list:

                # glob無法透過檔名直接過濾, 所以查出全部model再過濾
                if tokenizer_name == os.path.join(path, _robot_id + "+" + m_id + ".pickle"):
                    with open(tokenizer_name, 'rb') as handle:
                        self.tokenizer_dict.update({os.path.basename(tokenizer_name): pickle.load(handle)})

        self.logger.debug("tokenizer_dict : ")
        self.logger.debug(self.tokenizer_dict)

        api_end = datetime.now()
        api_duration = api_end - api_start
        self.logger.debug(api_duration)
        self.logger.debug('============load_model over=============')

    def predict(self, _robot_id, _model_id_list, _df_all=None):
        """
        結果預測
        多句話 >> 多個robot
        :param _robot_id:
        :param _model_id_list:
        :param _df_all:
        :return:
        """

        self.logger.debug("=== predict_test ===")
        empty_model_predict_df = pd.DataFrame({'sentence': [],
                                               'y_predict': [],
                                               'y_predict_id': [],
                                               'y_predict_name': [],
                                               'y_predict_probability': []})

        # 每次預測都重新load_model
        self.model_dict = {}
        self.tokenizer_dict = {}
        self.graph_dict = {}
        self.session_dict = {}
        self.load_model(_robot_id=_robot_id,
                        _model_id_list=_model_id_list)

        # 都空的就不預測
        if not bool(self.model_dict):
            self.logger.warning('There is no model in model_dict. Please reload model first.'.format(_robot_id))
            result = {
                "robot_id": _robot_id,
                "model_predict_df": empty_model_predict_df,
                "mapping": [],
                "mapping_name": []
            }
            result_list = [result]
            return result_list

        # filter by model_id_list
        train_log_df = HelperIntentModelUtility.query_train_log(robot_id=_robot_id)
        train_log_df = train_log_df[train_log_df['model_id'].isin(_model_id_list)]

        # self.logger.debug("train_log_df")
        # self.logger.debug(train_log_df)

        df_all = _df_all
        result_list = []

        # 多個模型
        for index, row in train_log_df.iterrows():

            algorithm = row['algorithm']
            algorithm_type = row['algorithm_type']
            algorithm_param = json.loads(row['algorithm_param'])
            model_id = row['model_id']
            nfs_model_id = row['nfs_model_id']
            nfs_tokenizer_id = row['nfs_tokenizer_id']
            mapping = row['mapping']
            mapping_name = row['mapping_name'].split(",")

            # optional
            sentence_max_len = algorithm_param.get('sentence_max_len', None)

            # self.logger.debug(mapping)
            # self.logger.debug(mapping_name)

            # 演算法工廠
            if algorithm == "textcnn":
                factory = IntentTextCnnModelFactory()

            elif algorithm == "1dcnn":
                factory = IntentOnedCnnModelFactory()

            elif algorithm == "2dcnn":
                factory = IntentTwodCnnModelFactory()

            elif algorithm == "gru":
                factory = IntentGRUModelFactory()

            elif algorithm == "rf":
                # randomforest
                factory = IntentRandomForestModelFactory()

            elif algorithm == "lg":
                # logisticregression
                factory = IntentLogisticRegressionModelFactory()

            # elif algorithm == "bert":
            #     # bert
            #     factory = IntenBertModelFactory()

            else:
                result = {
                    "robot_id": _robot_id,
                    "model_id": model_id,
                    "model_predict_df": empty_model_predict_df,
                    "mapping": [],
                    "mapping_name": []
                }
                result_list.append(result)
                continue

            self.logger.debug("nfs_model_id = {}; tokenizer_id = {}".format(nfs_model_id, nfs_tokenizer_id))

            fs_model = self.model_dict.get(nfs_model_id, None)
            fs_tokenizer = self.tokenizer_dict.get(nfs_tokenizer_id, None)

            if fs_model is None:
                self.logger.error('There is no model in disk with robot_id = {}. Please training model first.'.format(_robot_id))
                result = {
                    "robot_id": _robot_id,
                    "model_id": model_id,
                    "model_predict_df": empty_model_predict_df,
                    "mapping": [],
                    "mapping_name": []
                }
                result_list.append(result)
                continue

            # if tokenizer is None:
            #     self.logger.debug('There is no tokenozer in disk with robot_id = {}. Please training tokenozer first.'.format(_robot_id))
            #     result = {
            #         "robot_id": _robot_id,
            #         "model_id": model_id,ｘ
            #         "model_predict_df": empty_model_predict_df,
            #         "mapping": [],
            #         "mapping_name": []
            #     }
            #     result_list.append(result)
            #     continue

            model = factory.create_model()

            # 把之前訓練的model塞回去
            model.model = fs_model
            model.transformer = fs_tokenizer
            model.mapping = mapping
            model.mapping_name = mapping_name

            sentence_column = 'sentence'
            preprocessing_column = 'cut_words'
            feature_column = 'feature'
            # target_column = 'skill_id'
            # target_index_column = 'target_index'

            # 資料前處理
            sentence_df = model.preprocessing(sentence_df=df_all,
                                              input_column=sentence_column,
                                              output_column=preprocessing_column)

            # 特徵工程
            sentence_df = model.feature_engineering(sentence_df=sentence_df,
                                                    input_column=preprocessing_column,
                                                    output_column=feature_column,
                                                    model_param=algorithm_param)

            self.logger.debug('sentence_df')
            self.logger.debug(sentence_df)

            """
            在預測時用這個load_model時所建立的graph作為default graph
            """
            KS.set_session(self.session_dict[nfs_model_id])
            with self.graph_dict[nfs_model_id].as_default():

                predict_df = model.predict_result(test_df=sentence_df,
                                                  sentence_column=sentence_column,
                                                  feature_column=feature_column,
                                                  model_param=algorithm_param)

            # predict_df = model.predict_result(test_df=sentence_df,
            #                                   sentence_column=sentence_column,
            #                                   feature_column=feature_column)

            self.logger.debug('====== predict_df ====')
            self.logger.debug(predict_df)

            result = {
                "robot_id": _robot_id,
                "model_id": model_id,
                "model_predict_df": predict_df,
                "mapping": mapping,
                "mapping_name": mapping_name
            }
            result_list.append(result)

        # self.logger.debug(result_list)
        return result_list
