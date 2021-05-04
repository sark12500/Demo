# encoding=UTF-8
# !flask/bin/python

import os
import pandas as pd
import numpy as np
import uuid
import shutil
from datetime import datetime, timedelta, date
from sklearn import preprocessing
from sklearn.utils import shuffle
import logging
import xlsxwriter
from MssqlDAO import MssqlDAO
from MyEnv import Get_MyEnv
from Config_Helper import Get_HelperConfig
from Config_Format import Get_FormatConfig
from HelperDataApi import HelperDataApi


class HelperIntentModelUtility:

    def __init__(self):
        pass

    # MSSQL_DB = Get_MyEnv().env_helper_keyspace
    MSSQL_DB = Get_MyEnv().env_mssql_db

    config = Get_HelperConfig()
    HELPER_MASTER_TABLE = config.HELPER_MASTER_TABLE
    INTENT_MODEL = config.HELPER_INTENT_MODEL_TABLE
    INTENT_TRAIN_SENTENCE_SET = config.HELPER_INTENT_TRAIN_SENTENCE_SET_TABLE
    INTENT_TRAIN_SENTENCE = config.HELPER_INTENT_TRAIN_SENTENCE_TABLE
    INTENT_TRAIN_SENTENCE_CORPUS = config.HELPER_INTENT_TRAIN_SENTENCE_CORPUS_TABLE
    INTENT_TRAIN_SENTENCE_LOG = config.HELPER_INTENT_TRAIN_SENTENCE_LOG_TABLE
    INTENT_TRAIN_LOG = config.HELPER_INTENT_TRAIN_LOG_TABLE
    INTENT_TEST_LOG = config.HELPER_INTENT_TEST_LOG_TABLE
    INTENT_ALGORITHM = config.HELPER_INTENT_ALGORITHM_TABLE
    SENTENCE_MODIFY_LOG = config.HELPER_SENTENCE_MODIFY_LOG_TABLE
    SEMANTIC_MODEL_LOG = config.HELPER_SEMANTIC_MODEL_LOG_TABLE
    ROBOT_SKILLS = config.HELPER_ROBOT_SKILLS_TABLE
    ROBOT_AUTH = config.HELPER_ROBOT_AUTH_TABLE

    config = Get_FormatConfig()
    DATE_FORMAT_NORMAL = config.DATE_FORMAT_NORMAL
    DATE_FORMAT_YMD = config.DATE_FORMAT_YMD

    # dao = envCassandraDAO(Get_MyEnv().env_cassandra)
    dao = MssqlDAO(ip=Get_MyEnv().env_mssql_ip,
                   account=Get_MyEnv().env_mssql_account,
                   password=Get_MyEnv().env_mssql_password)

    # 自訂log
    logger = logging.getLogger('HelperIntentModelUtility.py')
    logger.setLevel(logging.DEBUG)

    @classmethod
    def get_uuid(cls):
        return str(uuid.uuid4())

    @classmethod
    def exec_cql_transations(cls, cql_list):

        cls.dao.execCQLTransaction(cls.MSSQL_DB, cql_list)

    @classmethod
    def exec_cql_transations_param_tuple(cls, cql_list, param_tuple_list):

        cls.dao.execCQLTransactionParamTuple(cls.MSSQL_DB, cql_list, param_tuple_list)

    @classmethod
    def query_user_robot(cls, camp_user_id):
        """
        查詢 camp_user_id 有編輯權限的機器人
        :param camp_user_id:
        :return:
        """

        helper_api = HelperDataApi()
        response = helper_api.query_robot(camp_user_id=camp_user_id)
        cls.logger.debug(response.status_code)
        if response.status_code == 200:
            cls.logger.debug("query_robot success ~")
            data_list = response.json()['data']

            data_df = pd.DataFrame(data_list)
            cls.logger.debug(data_df)

            train_log_pd = cls.query_train_log()
            train_log_pd = train_log_pd[['robot_id', 'model_id', 'create_date', 'create_user']]
            train_log_pd = train_log_pd.sort_values(by=['create_date'], ascending=False)
            # train_log_pd

            result = []
            for index, row in data_df.iterrows():

                robot_pd = train_log_pd[train_log_pd['robot_id'] == row['robot_id']].reset_index(drop=True)
                if len(robot_pd) > 0:
                    last_train_date = robot_pd['create_date'].iloc[0]
                    last_train_user = robot_pd['create_user'].iloc[0]
                else:
                    # cls.logger.debug('尚未訓練模型')
                    last_train_date = ''
                    last_train_user = ''

                # cls.logger.debug("robot_id : {}, last_train_date : {}, last_train_user : {}".format(row['robot_id'],
                #                                                                          last_train_date,
                #                                                                          last_train_user))

                # # TODO: 改成只query一次
                # train_log_pd = cls.query_train_log(row['robot_id'])
                # # cls.logger.debug(train_log_pd)
                # if len(train_log_pd) > 0:
                #     # new_log_pd = train_log_pd[0]
                #     last_train_date = train_log_pd['create_date'][0]
                #     last_train_user = train_log_pd['create_user'][0]
                # else:
                #     # cls.logger.debug('尚未訓練模型')
                #     last_train_date = ''
                #     last_train_user = ''

                user_robot_data = {
                    "robot_id": row['robot_id'],
                    "robot_name": row['robot_name'],
                    "robot_img": row['robot_img'],
                    "skill_count": len(row['skills']),
                    "last_train_date": last_train_date,
                    "last_train_user": last_train_user
                }
                result.append(user_robot_data)

            pd_df = pd.DataFrame(result)
            return pd_df

        else:
            cls.logger.debug("query_robot fail ~")
            return pd.DataFrame([])

    @classmethod
    def query_robot_skill(cls, robot_id, status='Y'):
        """
        查詢 robot_id 的 skill列表
        :param robot_id:
        :param status:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

        cql = "select * from " + cls.ROBOT_SKILLS + where_cql + ";"
        cls.logger.debug('query_robot_skill >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # # null補值, 以防新增欄位有null產生
        # fill_na_dict = {'enabled': 'Y',
        #                 'initial_status': 'Y',
        #                 'create_date': '',
        #                 'is_use': False,
        #                 'need_tokenizer': True
        #                 }
        # pd_df = pd_df.fillna(value=fill_na_dict)
        # pd_df['algorithm_param'] = pd_df['algorithm_param'].apply(lambda x: x if x else [])

        pd_df = pd_df.sort_values(by=['robot_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def query_algorithm(cls, algorithm_id=None):
        """
        查詢model
        :param algorithm_id:
        :return:
        """
        where_cql = ""
        if algorithm_id:
            algorithm_id = algorithm_id.lower()
            where_cql = where_cql + " where algorithm_id = '{algorithm_id}'".format(algorithm_id=algorithm_id)

        cql = "select * from " + cls.INTENT_ALGORITHM + where_cql + ";"
        cls.logger.debug('query_algorithm >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'algorithm_name': '',
                        'algorithm_type': '',
                        'create_user': 'camp',
                        'create_date': '',
                        'is_use': False,
                        'need_tokenizer': True
                        }
        pd_df = pd_df.fillna(value=fill_na_dict)
        pd_df['algorithm_param'] = pd_df['algorithm_param'].apply(lambda x: x if x else [])

        pd_df = pd_df.sort_values(by=['algorithm_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def query_model(cls, robot_id=None):
        """
        查詢model
        :param robot_id:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

        cql = "select * from " + cls.INTENT_MODEL + where_cql + ";"
        cls.logger.debug('query_model >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'min_confidence': 0.9,
                        'create_user': 'camp',
                        'create_date': '',
                        'modify_user': 'camp',
                        'modify_date': '',
                        'new_model_id': ''
                        }
        pd_df = pd_df.fillna(value=fill_na_dict)

        # min_confidence 轉回float
        pd_df['min_confidence'] = pd_df['min_confidence'].astype('float')

        pd_df = pd_df.sort_values(by=['robot_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def insert_model(cls, robot_id, modify_user='camp', min_confidence=0.9, new_model_id=''):
        """
        插入模型
        :param robot_id:
        :param modify_user:
        :param min_confidence:
        :param new_model_id:
        :return:
        """
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        cql = ("insert into {table}(" +
               "robot_id, min_confidence, new_model_id, " +
               "create_user, create_date, modify_user, modify_date)" +
               " values('{robot_id}', '{min_confidence}', '{new_model_id}'," +
               "'{create_user}', '{create_date}', '{modify_user}', '{modify_date}');")
        cql = cql.format(table=cls.INTENT_MODEL,
                         robot_id=robot_id,
                         min_confidence=str(min_confidence),
                         new_model_id=new_model_id,
                         create_user=modify_user,
                         create_date=modify_date,
                         modify_user=modify_user,
                         modify_date=modify_date
                         )

        cls.logger.debug('insert_model >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def update_model(cls, robot_id, modify_user, min_confidence, new_model_id):
        """
        更新模型
        :param robot_id:
        :param modify_user:
        :param min_confidence:
        :param new_model_id:
        :return:
        """
        # pdf = cls.query_model(robot_id)

        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        # def change_column_value(row):
        #     """
        #     只能有一個model上線, status='Y'
        #     """
        #     _robot_id = row['robot_id']
        #     _modify_user = row['modify_user']
        #     _modify_date = row['modify_date']
        #     _min_confidence = row['min_confidence']
        #
        #     if row['model_id'] == model_id:
        #
        #         _modify_user = modify_user
        #         _modify_date = modify_date
        #         _min_confidence = min_confidence
        #         _status = status
        #         _has_new_model = has_new_model
        #
        #     else:
        #         # Y的改成N
        #         if status == 'Y':
        #             _status = 'N'
        #
        #     return pd.Series([_robot_id, _model_id, _modify_user, _modify_date, _min_confidence, _status, _has_new_model])

        # pdf[['robot_id', 'model_id', 'modify_user', '_modify_date',
        #      'min_confidence', 'status', 'has_new_model']] = pdf.apply(change_column_value, axis=1)

        # for index, row in pdf.iterrows():
        #     cql = ("insert into {table}(" +
        #            "robot_id, model_id, min_confidence, status, has_new_model, " +
        #            "modify_user, modify_date)" +
        #            " values('{robot_id}', '{model_id}', '{min_confidence}', '{status}', {has_new_model},  " +
        #            "'{modify_user}', '{modify_date}');")
        #     cql = cql.format(table=cls.INTENT_MODEL,
        #                      robot_id=row['robot_id'],
        #                      model_id=row['model_id'],
        #                      min_confidence=str(row['min_confidence']),
        #                      status=row['status'],
        #                      has_new_model=row['has_new_model'],
        #                      modify_user=row['modify_user'],
        #                      modify_date=row['modify_date']
        #                      )
        #     cls.logger.debug('update_model >>> : ' + cql)
        #     cls.dao.execCQL(cls.HELPER_KEYSPACE, cql)

        cql = ("update {table}" +
               " set min_confidence={min_confidence}, new_model_id='{new_model_id}', " +
               " modify_user='{modify_user}', modify_date='{modify_date}'"
               " where robot_id='{robot_id}'"
               ).format(table=cls.INTENT_MODEL,
                        robot_id=robot_id,
                        min_confidence=str(min_confidence),
                        new_model_id=new_model_id,
                        modify_user=modify_user,
                        modify_date=modify_date
                        )

        cls.logger.debug('update_model >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

        if modify_user:
            cls.modify_model_record(robot_id, modify_user)

    @classmethod
    def update_model_by_column(cls, robot_id, modify_user=None,
                               min_confidence=None, new_model_id=None):
        """
        更新模型
        :param robot_id:
        :param modify_user:
        :param min_confidence:
        :param new_model_id:
        :return:
        """
        # pdf = cls.query_model(robot_id)

        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        columns_cql = ''
        if min_confidence:
            columns_cql = columns_cql + "min_confidence={min_confidence}, "

        if new_model_id is not None:
            columns_cql = columns_cql + "new_model_id='{new_model_id}', "

        if modify_user:
            modify_user = modify_user.lower()
            columns_cql = columns_cql + "modify_user='{modify_user}', "

        cql = ("update {table}" +
               " set " + columns_cql +
               " modify_date='{modify_date}'"
               " where robot_id='{robot_id}'"
               ).format(table=cls.INTENT_MODEL,
                        robot_id=robot_id,
                        min_confidence=str(min_confidence),
                        new_model_id=new_model_id,
                        modify_user=modify_user,
                        modify_date=modify_date
                        )

        cls.logger.debug('update_model_by_column >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

        if modify_user:
            cls.modify_model_record(robot_id, modify_user)

    @classmethod
    def update_model_list_by_column(cls, robot_id_list, modify_user=None,
                                    min_confidence=None, new_model_id=None):
        """
        更新模型
        :param robot_id_list:
        :param modify_user:
        :param min_confidence:
        :param new_model_id:
        :return:
        """
        # pdf = cls.query_model(robot_id)

        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        columns_cql = ''
        if min_confidence:
            columns_cql = columns_cql + "min_confidence={min_confidence}, "

        if new_model_id is not None:
            columns_cql = columns_cql + "new_model_id='{new_model_id}', "

        if modify_user:
            modify_user = modify_user.lower()
            columns_cql = columns_cql + "modify_user='{modify_user}', "

        for r_id in robot_id_list:
            cql = ("update {table}" +
                   " set " + columns_cql +
                   " modify_date='{modify_date}'"
                   " where robot_id='{robot_id}'"
                   ).format(table=cls.INTENT_MODEL,
                            robot_id=r_id,
                            min_confidence=str(min_confidence),
                            new_model_id=new_model_id,
                            modify_user=modify_user,
                            modify_date=modify_date
                            )
            cls.logger.debug('update_model_by_column >>> : ' + cql)
            cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def delete_model(cls, robot_id, modify_user=None):
        """
        刪除 model
        :param robot_id:
        :param modify_user:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()

            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

        cql = "delete from {table}".format(table=cls.INTENT_MODEL)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_model >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def modify_model_record(cls, robot_id, modify_user):
        """
        修改model紀錄
        :param robot_id:
        :param modify_user:
        :return:
        """

        robot_id = robot_id.lower()
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        cql = ("update {table}" +
               " set " +
               " modify_user='{modify_user}', modify_date='{modify_date}'"
               " where robot_id='{robot_id}'"
               ).format(table=cls.INTENT_MODEL,
                        robot_id=robot_id,
                        modify_user=modify_user,
                        modify_date=modify_date
                        )

        cls.logger.debug('modify_model_record >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_train_sentence_set(cls, robot_id, sentence_set_id=None):
        """
        查詢 train_sentence_set
        :param robot_id:
        :param sentence_set_id:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if sentence_set_id:
                sentence_set_id = sentence_set_id.lower()
                where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
                    sentence_set_id=sentence_set_id)

        cql = "select * from " + cls.INTENT_TRAIN_SENTENCE_SET + where_cql + ";"
        cls.logger.debug('query_train_sentence_set >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'create_user': 'camp',
                        'create_date': '',
                        'modify_user': 'camp',
                        'modify_date': '',
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        pd_df = pd_df.sort_values(by=['robot_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def insert_train_sentence_set(cls, robot_id, sentence_set_id, modify_user):
        """
        插入train_sentence_set
        :param robot_id:
        :param sentence_set_id:
        :param modify_user:
        :return:
        """
        robot_id = robot_id.lower()
        sentence_set_id = sentence_set_id.lower()
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        create_user = modify_user
        create_date = modify_date

        cql = ("insert into {table}(" +
               "robot_id, sentence_set_id, create_user, create_date, modify_user, modify_date) "
               "values('{robot_id}', '{sentence_set_id}', "
               "'{create_user}', '{create_date}', '{modify_user}', '{modify_date}');")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE_SET,
                         robot_id=robot_id,
                         sentence_set_id=sentence_set_id,
                         create_user=create_user,
                         create_date=create_date,
                         modify_user=modify_user,
                         modify_date=modify_date
                         )

        cls.logger.debug('insert_train_sentence_set >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def update_train_sentence_set(cls, robot_id, sentence_set_id, modify_user):
        """
        更新 train_sentence_set
        :param robot_id:
        :param sentence_set_id:
        :param modify_user:
        :return:
        """
        robot_id = robot_id.lower()
        sentence_set_id = sentence_set_id.lower()
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        cql = ("update {table}" +
               " set " +
               " modify_user='{modify_user}', modify_date='{modify_date}'"
               " where robot_id='{robot_id}' and sentence_set_id='{sentence_set_id}'"
               ).format(table=cls.INTENT_TRAIN_SENTENCE_SET,
                        robot_id=robot_id,
                        sentence_set_id=sentence_set_id,
                        modify_user=modify_user,
                        modify_date=modify_date
                        )

        cls.logger.debug('update_train_sentence_set >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def delete_train_sentence_set(cls, robot_id, sentence_set_id_list=[], modify_user=None):
        """
        刪除 train_sentence_set
        :param robot_id:
        :param sentence_set_id_list:
        :param modify_user:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if sentence_set_id_list:
                cql_in = "'"
                for index, value in enumerate(sentence_set_id_list):
                    cql_in = cql_in + value + "'"
                    if index + 1 != len(sentence_set_id_list):
                        cql_in = cql_in + ", '"

                where_cql = where_cql + " and sentence_set_id in ({cql_in})".format(cql_in=cql_in)

            # if sentence_set_id:
            #     sentence_set_id = sentence_set_id.lower()
            #     where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
            #         sentence_set_id=sentence_set_id)

        cql = "delete from {table}".format(table=cls.INTENT_TRAIN_SENTENCE_SET)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_train_sentence_set >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_train_sentence(cls, robot_id, sentence_set_id=None, skill_id=None, sentence_id=None):
        """
        查詢model
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :param sentence_id:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if sentence_set_id:
                sentence_set_id = sentence_set_id.lower()
                where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
                    sentence_set_id=sentence_set_id)

                if skill_id:
                    skill_id = skill_id.lower()
                    where_cql = where_cql + " and skill_id = '{skill_id}'".format(skill_id=skill_id)

                    if sentence_id:
                        where_cql = where_cql + " and sentence_id = {sentence_id}".format(sentence_id=sentence_id)

        cql = "select * from " + cls.INTENT_TRAIN_SENTENCE + where_cql + ";"
        cls.logger.debug('query_train_sentence >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'sentence': '',
                        'create_date': ''
                        }
        pd_df = pd_df.fillna(value=fill_na_dict)
        pd_df = pd_df.sort_values(by=['robot_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def query_train_sentence_count(cls, robot_id, sentence_set_id=None, skill_id=None):
        """
        查詢句子數量
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if sentence_set_id:
                sentence_set_id = sentence_set_id.lower()
                where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
                    sentence_set_id=sentence_set_id)

                if skill_id:
                    skill_id = skill_id.lower()
                    where_cql = where_cql + " and skill_id = '{skill_id}'".format(skill_id=skill_id)

        cql = "select count(*) as count from " + cls.INTENT_TRAIN_SENTENCE + where_cql + ";"
        cls.logger.debug('query_train_sentence_count >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)
        count = pd_df['count'][0]
        # cls.logger.debug(type(count))
        # cls.logger.debug(type(int(count)))
        return count

    @classmethod
    def insert_train_sentence(cls, robot_id, sentence_set_id, skill_id, sentence):
        """
        插入訓練句子 by sentence_set
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :param sentence:
        :return:
        """
        # create_date = str(datetime.now())
        create_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        sentence_id = cls.get_uuid()

        # cql = ("insert into {table}(" +
        #        "robot_id, sentence_set_id, skill_id, sentence_id, sentence, create_date) "
        #        " values('{robot_id}', '{sentence_set_id}', '{skill_id}', '{sentence_id}', '{sentence}', '{create_date}');")
        # cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE,
        #                  robot_id=robot_id,
        #                  sentence_set_id=sentence_set_id,
        #                  skill_id=skill_id,
        #                  sentence_id=sentence_id,
        #                  sentence=sentence,
        #                  create_date=create_date
        #                  )
        #
        # cls.logger.debug('insert_train_sentence >>> : ' + cql)
        # cls.dao.execCQL(cls.MSSQL_DB, cql)

        cql = ("insert into {table} values (%s, %s, %s, %s, %s, %s)")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE)
        param_tuple = (robot_id, sentence_set_id, skill_id, sentence_id, create_date, sentence)

        cls.logger.debug('insert_train_sentence >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))
        cls.dao.execCQLParamTuple(cls.MSSQL_DB, cql, param_tuple)

    @classmethod
    def insert_train_sentence_get_cql(cls, robot_id, sentence_set_id, skill_id, sentence):
        """
        插入訓練句子 by sentence_set 的sql語法與參數
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :param sentence:
        :return:
        """
        # create_date = str(datetime.now())
        create_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        sentence_id = cls.get_uuid()
        # cql = ("insert into {table}(" +
        #        "robot_id, sentence_set_id, skill_id, sentence_id, sentence, create_date) "
        #        " values('{robot_id}', '{sentence_set_id}', '{skill_id}', '{sentence_id}', '{sentence}', '{create_date}');")
        # cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE,
        #                  robot_id=robot_id,
        #                  sentence_set_id=sentence_set_id,
        #                  skill_id=skill_id,
        #                  sentence_id=sentence_id,
        #                  sentence=sentence,
        #                  create_date=create_date
        #                  )
        #
        # cls.logger.debug('insert_train_sentence >>> : ' + cql)
        # # cls.dao.execCQL(cls.MSSQL_DB, cql)

        cql = ("insert into {table} values (%s, %s, %s, %s, %s, %s)")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE)
        param_tuple = (robot_id, sentence_set_id, skill_id, sentence_id, create_date, sentence.decode('utf-8'))

        cls.logger.debug('insert_train_sentence >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))

        return cql, param_tuple

    @classmethod
    def insert_sentence_file(cls, robot_id, sentence_set_id, file, file_type, save_file_name, save_path, sentence_df,
                             modify_user=''):
        """
        檔案寫入file server
        :param robot_id:
        :param sentence_set_id:
        :param file:
        :param save_file_name:
        :param save_path:
        :param modify_user:
        :param sentence_df:
        :return:
        """

        robot_id = robot_id.lower()
        # cls.logger.debug(os.path.isdir(save_path + robot_id))
        # cls.logger.debug(save_path + robot_id)
        if os.path.isdir(save_path + robot_id) == False:
            os.mkdir(save_path + robot_id)

        # cls.logger.debug(os.path.isdir(save_path + robot_id + "/" + sentence_set_id))
        # cls.logger.debug(save_path + robot_id + "/" + sentence_set_id)

        if os.path.isdir(save_path + robot_id + "/" + sentence_set_id) == False:
            os.mkdir(save_path + robot_id + "/" + sentence_set_id)

        sentence_file_path = save_path + robot_id + "/" + sentence_set_id + "/" + save_file_name
        cls.logger.debug("sentence_file_path : {}".format(sentence_file_path))
        cls.logger.debug(file_type)
        # cls.logger.debug(len(file.read()))
        # file.save(sentence_file_path) # <<< file will equal zero byte !!!

        if file_type in ['csv']:
            cls.logger.debug('in csv')
            sentence_df.to_csv(sentence_file_path, index=False)
        else:
            cls.logger.debug('in excel')
            """
            如果發生無法使用to_excel, 請在該臺環境安裝 pip install openpyxl
            """
            sentence_df.to_excel(sentence_file_path, index=False, engine='xlsxwriter')

        # # 修改header modify資訊
        # if modify_user:
        #     cls.modify_sentence_record(robot_id, sentence_set_id, modify_user)

        # # save local
        # cls.logger.debug("os.path.isdir(save_filename_path) = {}".format(os.path.isdir(save_filename_path)))
        # if not os.path.isdir(Get_MyEnv().env_file_server_path + 'instance_item_upload/' + modify_user):
        #     os.mkdir(Get_MyEnv().env_file_server_path + 'instance_item_upload/' + modify_user)
        # cls.logger.debug("image_save_path : {}".format(_save_path + _save_file_name))
        # _file.save(_save_path + _save_file_name)
        #
        # full_filename = save_filename_path + modify_user + '-' + save_filename
        # logger.debug('full file name : {}'.format(full_filename))
        #
        # if convert_type == '.csv':
        #     instance_df.to_csv(full_filename, index=False)
        # else:
        #     instance_df.to_excel(full_filename, index=False)
        #
        # file.save(save_filename_path + '-' + save_filename)
        #
        # upload_df = pd.read_excel(file.filename)

        # # 保存檔案 - 使用絕對路徑
        # save_filename = file.filename
        #
        # if Get_MyEnv().env_file == Get_MyEnv().LOCAL:
        #     # local
        #     from_save_filename_path = Get_MyEnv().env_file_path + 'instance_item/' + modify_user + '-' + save_filename
        #     to_save_filename_path = Get_MyEnv().env_file_path + 'instance_item/' + modify_user + '-' + save_filename
        # else:
        #     # production
        #     from_save_filename_path = Get_MyEnv().env_file_abs_path + "FlaskApi_HelperData/app/file/" + \
        #                               'instance_item/' + modify_user + '-' + save_filename
        #     to_save_filename_path = Get_MyEnv().env_file_abs_path + "FlaskApi_HelperData/app/file/" + \
        #                             'instance_item/' + modify_user + '-' + save_filename
        #
        # logger.debug('file name : {}'.format(save_filename))
        # logger.debug('from_save_filename_path : {}'.format(from_save_filename_path))
        # logger.debug('to_save_filename_path : {}'.format(to_save_filename_path))
        #
        # # form_data file to all low-balance node
        # # file.save(save_filename_path)
        # # if not os.path.isdir(save_filename_path):
        # #     os.mkdir(save_filename_path)
        #
        # if convert_type == '.csv':
        #     instance_df.to_csv(from_save_filename_path, index=False)
        # else:
        #     # xlsx, xls
        #     instance_df.to_excel(from_save_filename_path, index=False)
        #
        # # 將檔案複製到其他節點
        # result = copy_file_ssh(from_save_filename_path, to_save_filename_path)
        #
        # if result == False:
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    message="save file to disk error !!!",
        #                    data=[],
        #                    ), 999

    @classmethod
    def export_sentence_file(cls, robot_id, sentence_set_id, file_type,
                             save_file_name, save_path, sentence_df):
        """
        檔案匯出file server
        :param robot_id:
        :param sentence_set_id:
        :param file_type:
        :param save_file_name:
        :param save_path:
        :param sentence_df:
        :return:
        """

        robot_id = robot_id.lower()
        cls.logger.debug(os.path.isdir(save_path + robot_id))
        cls.logger.debug(save_path + robot_id)
        if os.path.isdir(save_path + robot_id) == False:
            os.mkdir(save_path + robot_id)

        cls.logger.debug(os.path.isdir(save_path + robot_id + "/" + sentence_set_id))
        cls.logger.debug(save_path + robot_id + "/" + sentence_set_id)

        if os.path.isdir(save_path + robot_id + "/" + sentence_set_id) == False:
            os.mkdir(save_path + robot_id + "/" + sentence_set_id)

        sentence_file_path = save_path + robot_id + "/" + sentence_set_id + "/" + save_file_name
        save_path_final = robot_id + "/" + sentence_set_id + "/" + save_file_name
        cls.logger.debug("sentence_file_path : {}".format(sentence_file_path))
        cls.logger.debug(file_type)
        # cls.logger.debug(len(file.read()))
        # file.save(sentence_file_path) # <<< file will equal zero byte !!!

        unicode_names = [u'意圖', u'語句']
        str_names = ['意圖', '語句']
        if file_type in ['csv']:
            cls.logger.debug('in csv')
            sentence_df.columns = str_names
            #         sentence_df.columns = unicode_names
            #         cls.logger.debug(sentence_df)
            save_path_final = save_path_final + '.csv'
            sentence_df.to_csv(save_path_final, index=False, encoding='utf_8_sig')
        else:
            cls.logger.debug('in excel')
            """
            如果發生無法使用to_excel, 請在該臺環境安裝 pip install openpyxl
            """
            #         sentence_df.columns = str_names
            sentence_df.columns = unicode_names
            save_path_final = save_path_final + '.xlsx'
            sentence_df.to_excel(sentence_file_path + '.xlsx', index=False, encoding='utf_8_sig', engine='xlsxwriter')

        return save_path_final

    @classmethod
    def update_train_sentence(cls, robot_id, sentence_set_id, skill_id, sentence_id, sentence):
        """
        更新訓練語句
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :param sentence_id:
        :param sentence:
        :return:
        """

        # cql = ("update {table}" +
        #        " set " +
        #        " sentence='{sentence}'"
        #        " where robot_id='{robot_id}' and sentence_set_id='{sentence_set_id}'" +
        #        " and skill_id='{skill_id}'" +
        #        " and sentence_id='{sentence_id}'"
        #        ).format(table=cls.INTENT_TRAIN_SENTENCE,
        #                 robot_id=robot_id,
        #                 sentence_set_id=sentence_set_id,
        #                 skill_id=skill_id,
        #                 sentence_id=sentence_id,
        #                 sentence=sentence
        #                 )
        #
        # cls.logger.debug('update_train_sentence >>> : ' + cql)
        # cls.dao.execCQL(cls.MSSQL_DB, cql)

        cql = ("update {table} set sentence=%s where robot_id=%s and sentence_set_id=%s and skill_id=%s and "
               "sentence_id=%s")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE)
        param_tuple = (sentence, robot_id, sentence_set_id, skill_id, sentence_id)

        cls.logger.debug('insert_train_sentence >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))
        cls.dao.execCQLParamTuple(cls.MSSQL_DB, cql, param_tuple)

    @classmethod
    def delete_train_sentence(cls, robot_id, sentence_set_id=None, skill_id=None, sentence_list=None, modify_user=None):
        """
        刪除 train_sentence
        :param robot_id:
        :param sentence_set_id:
        :param skill_id:
        :param sentence_list:
        :param modify_user:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if sentence_set_id:
                sentence_set_id = sentence_set_id.lower()
                where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
                    sentence_set_id=sentence_set_id)

                if skill_id:
                    skill_id = skill_id.lower()
                    where_cql = where_cql + " and skill_id = '{skill_id}'".format(skill_id=skill_id)

                    if sentence_list:
                        cql_in = ""
                        for index, value in enumerate(sentence_list):
                            cql_in = cql_in + value
                            if index + 1 != len(sentence_list):
                                cql_in = cql_in + "', '"

                        where_cql = where_cql + " and sentence_id in ('{cql_in}')".format(cql_in=cql_in)

        cql = "delete from {table}".format(table=cls.INTENT_TRAIN_SENTENCE)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_train_sentence >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_train_sentence_log(cls, robot_id, model_id=None):
        """
        查詢model
        :param robot_id:
        :param model_id:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id:
                model_id = model_id.lower()
                where_cql = where_cql + " and model_id = '{model_id}'".format(model_id=model_id)

        cql = "select * from " + cls.INTENT_TRAIN_SENTENCE_LOG + where_cql + ";"
        cls.logger.debug('query_train_sentence_log >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'sentence': ''
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        pd_df = pd_df.sort_values(by=['robot_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def insert_train_sentence_log(cls, robot_id, model_id, skill_id, sentence_id, sentence, cut_sentence, create_date):
        """
        插入訓練句子 by sentence_set
        :param robot_id:
        :param model_id:
        :param skill_id:
        :param sentence_id:
        :param sentence:
        :param cut_sentence:
        :param create_date:
        :return:
        """

        # cql = ("insert into {table}(" +
        #        "robot_id, model_id, skill_id, sentence_id, sentence, cut_sentence, create_date) "
        #        " values('{robot_id}', '{model_id}', '{skill_id}', '{sentence_id}', '{sentence}', '{cut_sentence}', '{create_date}');")
        # cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE_LOG,
        #                  robot_id=robot_id,
        #                  model_id=model_id,
        #                  skill_id=skill_id,
        #                  sentence_id=sentence_id,
        #                  sentence=sentence,
        #                  cut_sentence=cut_sentence,
        #                  create_date=create_date
        #                  )
        #
        # cls.logger.debug('insert_train_sentence_log >>> : ' + cql)
        # cls.dao.execCQL(cls.MSSQL_DB, cql)

        cql = ("insert into {table} values (%s, %s, %s, %s, %s, %s, %s)")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE_LOG)
        param_tuple = (robot_id, model_id, skill_id, sentence_id, create_date, cut_sentence, sentence)

        cls.logger.debug('insert_train_sentence >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))
        cls.dao.execCQLParamTuple(cls.MSSQL_DB, cql, param_tuple)


    @classmethod
    def insert_train_sentence_log_get_cql(cls, robot_id, model_id, skill_id, sentence_id, sentence, cut_sentence, create_date):
        """
        插入訓練句子 by sentence_set
        :param robot_id:
        :param model_id:
        :param skill_id:
        :param sentence_id:
        :param sentence:
        :param cut_sentence:
        :param create_date:
        :return:
        """
        # cql = ("insert into {table}(" +
        #        "robot_id, model_id, skill_id, sentence_id, sentence, cut_sentence, create_date) "
        #        " values('{robot_id}', '{model_id}', '{skill_id}', '{sentence_id}', '{sentence}', '{cut_sentence}', '{create_date}');")
        # cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE_LOG,
        #                  robot_id=robot_id,
        #                  model_id=model_id,
        #                  skill_id=skill_id,
        #                  sentence_id=sentence_id,
        #                  sentence=sentence,
        #                  cut_sentence=cut_sentence,
        #                  create_date=create_date
        #                  )
        #
        # cls.logger.debug('insert_train_sentence_log_get_cql >>> : ' + cql)
        # # cls.dao.execCQL(cls.MSSQL_DB, cql)

        cql = ("insert into {table} values (%s, %s, %s, %s, %s, %s, %s)")
        cql = cql.format(table=cls.INTENT_TRAIN_SENTENCE_LOG)
        param_tuple = (robot_id, model_id, skill_id, sentence_id, create_date, cut_sentence, sentence)

        # cls.logger.debug('insert_train_sentence_log_get_cql >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))
        # cls.dao.execCQLParamTuple(cls.MSSQL_DB, cql, param_tuple)

        return cql, param_tuple

    @classmethod
    def delete_train_sentence_log(cls, robot_id, model_id_list=[], modify_user=None):
        """
        刪除 train_sentence_log
        :param robot_id:
        :param model_id_list:
        :param modify_user:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()
            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id_list:
                cql_in = "'"
                for index, value in enumerate(model_id_list):
                    cql_in = cql_in + value + "'"
                    if index + 1 != len(model_id_list):
                        cql_in = cql_in + ", '"

                where_cql = where_cql + " and model_id in ({cql_in})".format(cql_in=cql_in)

        cql = "delete from {table}".format(table=cls.INTENT_TRAIN_SENTENCE_LOG)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_train_sentence_log >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def modify_sentence_record(cls, robot_id, sentence_set_id, modify_user):
        """
        修改sentence紀錄
        :param robot_id:
        :param sentence_set_id:
        :param modify_user:
        :return:
        """

        robot_id = robot_id.lower()
        sentence_set_id = sentence_set_id.lower()
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        cql = ("update {table}" +
               " set " +
               " modify_user='{modify_user}', modify_date='{modify_date}'"
               " where robot_id='{robot_id}' and sentence_set_id='{sentence_set_id}'"
               ).format(table=cls.INTENT_TRAIN_SENTENCE_SET,
                        robot_id=robot_id,
                        sentence_set_id=sentence_set_id,
                        modify_user=modify_user,
                        modify_date=modify_date
                        )

        cls.logger.debug('modify_sentence_record >>> : ' + cql)
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_train_log(cls, robot_id=None, model_id=None, status=None):
        """
        查詢 query_train_log
        :param robot_id:
        :param model_id:
        :param status:
        :return:
        """

        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()

            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id:
                model_id = model_id.lower()
                where_cql = where_cql + " and model_id = '{model_id}'".format(model_id=model_id)

            if status:
                where_cql = where_cql + " and status = '{status}'".format(status=status)

        cql = "select * from " + cls.INTENT_TRAIN_LOG + where_cql + ";"
        cls.logger.debug('query_train_log >>> {} ; keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        pd_df = pd_df.fillna('')
        fill_na_dict = {'train_name': '',
                        'sentence_set_id': '',
                        'algorithm': '',
                        'algorithm_param': '',
                        'algorithm_type': '',
                        'tokenizer_id': '',
                        'nfs_model_id': '',
                        'nfs_tokenizer_id': '',
                        'train_test_size': '',
                        'mapping_name': '',
                        'create_user': 'camp',
                        'create_date': '',
                        'status': 'N'
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        if len(pd_df) > 0:

            def split_udf(row):

                if len(row['mapping']) == 0:
                    return []
                else:
                    return row['mapping'].split(",")

            pd_df['mapping'] = pd_df.apply(split_udf, axis=1)

        pd_df = pd_df.sort_values(by=['create_date'], ascending=False).reset_index(drop=True)
        return pd_df

    @classmethod
    def insert_train_log(cls, robot_id, model_id, sentence_set_id,
                         algorithm, algorithm_param, algorithm_type,
                         tokenizer_id, mapping, mapping_name,
                         status, nfs_model_id, nfs_tokenizer_id, train_test_size=0.9, modify_user='camp'):
        """
        插入訓練紀錄 insert_train_log
        :param robot_id:
        :param model_id:
        :param sentence_set_id:
        :param algorithm:
        :param algorithm_param:
        :param algorithm_type:
        :param tokenizer_id:
        :param mapping:
        :param mapping_name:
        :param status:
        :param train_test_size:
        :param modify_user:
        :param nfs_model_id:
        :param nfs_tokenizer_id:
        :return:
        """

        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        # # unicode to string
        # mapping = [s.encode('ascii') for s in mapping]
        # # mapping_name = [s.encode('ascii') for s in mapping_name]
        mapping = ",".join(mapping)
        mapping_name = ",".join(mapping_name)
        cql = ("insert into {table}(" +
               "robot_id, model_id, sentence_set_id, algorithm, algorithm_param, algorithm_type, " +
               "tokenizer_id, mapping, mapping_name, status, train_test_size, " +
               "create_user, create_date, nfs_model_id,  nfs_tokenizer_id)" +
               " values('{robot_id}', '{model_id}', '{sentence_set_id}', '{algorithm}', "
               "'{algorithm_param}', '{algorithm_type}', " +
               "'{tokenizer_id}', '{mapping}', N'{mapping_name}', '{status}', '{train_test_size}', "
               "'{create_user}', '{create_date}', '{nfs_model_id}', '{nfs_tokenizer_id}');")
        cql = cql.format(table=cls.INTENT_TRAIN_LOG,
                         robot_id=robot_id,
                         model_id=model_id,
                         sentence_set_id=sentence_set_id,
                         algorithm=algorithm,
                         algorithm_param=algorithm_param,
                         algorithm_type=algorithm_type,
                         tokenizer_id=tokenizer_id,
                         mapping=str(mapping),
                         mapping_name=mapping_name,
                         status=status,
                         train_test_size=str(train_test_size),
                         create_user=modify_user,
                         create_date=modify_date,
                         nfs_model_id=nfs_model_id,
                         nfs_tokenizer_id=nfs_tokenizer_id
                         )

        cls.logger.debug('insert_train_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def update_train_log(cls, robot_id, model_id, sentence_set_id,
                         algorithm, algorithm_param, algorithm_type,
                         tokenizer_id, mapping, mapping_name,
                         status, train_test_size=0.9, modify_user='camp'):
        """
        更新訓練紀錄 update_train_log
        :param robot_id:
        :param model_id:
        :param sentence_set_id:
        :param algorithm:
        :param algorithm_param:
        :param algorithm_type:
        :param tokenizer_id:
        :param mapping:
        :param mapping_name:
        :param status:
        :param train_test_size:
        :param modify_user:
        :return:
        """

        pdf = cls.query_train_log(robot_id)

        cls.logger.debug(pdf[['robot_id', 'model_id', 'mapping', 'mapping_name']])

        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        def change_column_value(row):
            """
            只能有一個model上線, status='Y'
            """
            _robot_id = row['robot_id']
            _model_id = row['model_id']
            _sentence_set_id = row['sentence_set_id']
            _algorithm = row['algorithm']
            _algorithm_param = row['algorithm_param']
            _algorithm_type = row['algorithm_type']
            _tokenizer_id = row['tokenizer_id']
            _mapping = row['mapping']
            _mapping_name = row['mapping_name']
            _train_test_size = row['train_test_size']
            _status = row['status']
            _create_user = row['create_user']
            _create_date = row['create_date']
            _modify_user = modify_user
            _modify_date = modify_date

            if row['model_id'] == model_id:
                cls.logger.debug('row[model_id] == model_id')

                _sentence_set_id = sentence_set_id
                _algorithm = algorithm
                _algorithm_param = algorithm_param
                _algorithm_type = algorithm_type
                _tokenizer_id = tokenizer_id
                _mapping = mapping
                _mapping_name = mapping_name
                _train_test_size = train_test_size

                _status = status

            else:
                # 如果要設定爲Y, 其他都改成N, 只能有一個Y
                # 如果要設定爲N, 其他都不動
                if status == 'Y':
                    _status = 'N'

            return pd.Series([_robot_id, _model_id, _modify_user, _modify_date, _create_user, _create_date,
                              _sentence_set_id, _algorithm, _algorithm_param, _algorithm_type,
                              _tokenizer_id, _mapping, _mapping_name, _train_test_size, _status])

        pdf[['robot_id', 'model_id', 'modify_user', 'modify_date', 'create_user', 'create_date',
             'sentence_set_id', 'algorithm', 'algorithm_param', 'algorithm_type',
             'tokenizer_id', 'mapping', 'mapping_name', 'train_test_size', 'status']] = pdf.apply(change_column_value,
                                                                                                  axis=1)

        cls.logger.debug(pdf[['robot_id', 'model_id', 'mapping', 'mapping_name']])

        for index, row in pdf.iterrows():

            # unicode to string
            # mapping = [s.encode('ascii') for s in row['mapping']]
            if type(row['mapping']) is list:
                mapping = ",".join(row['mapping'])

            if type(row['mapping_name']) is list:
                mapping_name = ",".join(row['mapping_name'])
                cls.logger.debug('change !!!')
                cls.logger.debug(row['model_id'])
                cls.logger.debug(mapping_name)

            cql = ("update {table}" +
                   " set " +
                   " sentence_set_id='{sentence_set_id}', algorithm='{algorithm}', " +
                   " algorithm_param='{algorithm_param}', algorithm_type='{algorithm_type}', " +
                   " tokenizer_id='{tokenizer_id}', mapping='{mapping}', " +
                   " mapping_name=N'{mapping_name}', status='{status}', " +
                   " train_test_size='{train_test_size}', " +
                   " create_user='{create_user}', create_date='{create_date}' " +
                   " where robot_id='{robot_id}' and model_id='{model_id}'"
                   ).format(table=cls.INTENT_TRAIN_LOG,
                            robot_id=row['robot_id'],
                            model_id=row['model_id'],
                            sentence_set_id=row['sentence_set_id'],
                            algorithm=row['algorithm'],
                            algorithm_param=row['algorithm_param'],
                            algorithm_type=row['algorithm_type'],
                            tokenizer_id=row['tokenizer_id'],
                            mapping=str(mapping),
                            mapping_name=mapping_name,
                            status=row['status'],
                            train_test_size=str(train_test_size),
                            create_user=row['create_user'],
                            create_date=row['create_date']
                            )

            cls.logger.debug('update_train_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
            cls.dao.execCQL(cls.MSSQL_DB, cql)

        # # unicode to string
        # mapping = [s.encode('ascii') for s in mapping]
        # cql = ("insert into {table}(" +
        #        "robot_id, model_id, sentence_set_id, algorithm, algorithm_param," +
        #        "sentence_max_len, tokenizer_id, mapping, status, train_test_size, create_user, create_date)" +
        #        "values('{robot_id}', '{model_id}', '{sentence_set_id}', '{algorithm}', "
        #        "'{algorithm_param}', " +
        #        "{sentence_max_len}, '{tokenizer_id}', {mapping}, '{status}', '{train_test_size}', '{create_user}', "
        #        "'{create_date}');")
        # cql = cql.format(table=cls.INTENT_TRAIN_LOG,
        #                  robot_id=robot_id,
        #                  model_id=model_id,
        #                  sentence_set_id=sentence_set_id,
        #                  algorithm=algorithm,
        #                  algorithm_param=algorithm_param,
        #                  sentence_max_len=sentence_max_len,
        #                  tokenizer_id=tokenizer_id,
        #                  mapping=str(mapping),
        #                  status=status,
        #                  train_test_size=str(train_test_size),
        #                  create_user=modify_user,
        #                  create_date=modify_date
        #                  )
        #
        # cls.logger.debug('update_train_log >>> : ' + cql)
        # cls.dao.execCQL(cls.HELPER_KEYSPACE, cql)

    @classmethod
    def update_train_log_by_column(cls, robot_id, model_id, modify_user=None, sentence_set_id=None,
                                   algorithm=None, algorithm_param=None, algorithm_type=None,
                                   tokenizer_id=None, mapping=None, mapping_name=None,
                                   status=None, train_test_size=None):
        """
        更新訓練紀錄 update_train_log_by_column
        :param robot_id:
        :param model_id:
        :param modify_user:
        :param sentence_set_id:
        :param algorithm:
        :param algorithm_param:
        :param algorithm_type:
        :param tokenizer_id:
        :param mapping:
        :param mapping_name:
        :param status:
        :param train_test_size:
        :return:
        """

        pdf = cls.query_train_log(robot_id)

        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        # cls.logger.debug(pdf[['robot_id', 'model_id', 'mapping', 'mapping_name']])

        def change_column_value(row):
            """
            只能有一個model上線, status='Y'
            """
            _robot_id = row['robot_id']
            _model_id = row['model_id']
            _sentence_set_id = row['sentence_set_id']
            _algorithm = row['algorithm']
            _algorithm_param = row['algorithm_param']
            _algorithm_type = row['algorithm_type']
            _tokenizer_id = row['tokenizer_id']
            _mapping = row['mapping']
            _mapping_name = row['mapping_name']
            _train_test_size = row['train_test_size']
            _status = row['status']
            _create_user = row['create_user']
            _create_date = row['create_date']
            _modify_user = modify_user
            _modify_date = modify_date

            if row['model_id'] == model_id:

                _sentence_set_id = sentence_set_id
                _algorithm = algorithm
                _algorithm_param = algorithm_param
                _algorithm_type = algorithm_type
                _tokenizer_id = tokenizer_id
                _mapping = mapping
                _mapping_name = mapping_name
                _train_test_size = train_test_size
                _status = status

            else:
                # 如果要設定爲Y, 其他都改成N, 只能有一個Y
                # 如果要設定爲N, 其他都不動
                if status == 'Y':
                    _status = 'N'

            return pd.Series([_robot_id, _model_id, _modify_user, _modify_date, _create_user, _create_date,
                              _sentence_set_id, _algorithm, _algorithm_param, _algorithm_type,
                              _tokenizer_id, _mapping, _mapping_name, _train_test_size, _status])

        pdf[['robot_id', 'model_id', 'modify_user', 'modify_date', 'create_user', 'create_date',
             'sentence_set_id', 'algorithm', 'algorithm_param', 'algorithm_type',
             'tokenizer_id', 'mapping', 'mapping_name', 'train_test_size', 'status']] = pdf.apply(change_column_value,
                                                                                                  axis=1)

        for index, row in pdf.iterrows():
            columns_cql = ''
            if sentence_set_id:
                columns_cql = columns_cql + "sentence_set_id={sentence_set_id}, "

            if algorithm:
                columns_cql = columns_cql + "algorithm='{algorithm}', "

            if algorithm_param:
                columns_cql = columns_cql + "algorithm_param='{algorithm_param}', "

            if algorithm_type:
                columns_cql = columns_cql + "algorithm_type='{algorithm_type}', "

            if tokenizer_id:
                columns_cql = columns_cql + "tokenizer_id='{tokenizer_id}', "

            if mapping:
                columns_cql = columns_cql + "mapping='{mapping}', "

            if mapping_name:
                columns_cql = columns_cql + "mapping_name=N'{mapping_name}', "

            if status:
                columns_cql = columns_cql + "status='{status}', "

            if train_test_size:
                columns_cql = columns_cql + "train_test_size='{train_test_size}', "

            if modify_user:
                modify_user = modify_user.lower()
                columns_cql = columns_cql + "create_user='{create_user}', "

            # # 轉成string list才能寫入DB
            # mapping_cql = '[]'
            # if row['mapping']:
            #     mapping_cql = str([s.encode('ascii') for s in row['mapping']])

            mapping_cql = ''
            if row['mapping']:
                mapping_cql = row['mapping']

                if type(row['mapping']) is list:
                    mapping_cql = ",".join(row['mapping'])

            mapping_name_cql = ''
            if row['mapping_name']:
                mapping_name_cql = row['mapping_name']

                if type(row['mapping_name']) is list:
                    mapping_name_cql = ",".join(row['mapping_name'])

            # mapping_name_cql = '[]'
            # cls.logger.debug(row['mapping_name'])
            # if row['mapping_name']:
            #     mapping_name_cql = str([s.encode('utf8') for s in row['mapping_name']])
            #     cls.logger.debug(mapping_name_cql)
            if not train_test_size:
                train_test_size = 0.9

            cql = ("update {table}" +
                   " set " + columns_cql +
                   " create_date='{create_date}'"
                   " where robot_id='{robot_id}' and model_id='{model_id}'"
                   ).format(table=cls.INTENT_TRAIN_LOG,
                            robot_id=row['robot_id'],
                            model_id=row['model_id'],
                            sentence_set_id=row['sentence_set_id'],
                            algorithm=row['algorithm'],
                            algorithm_param=row['algorithm_param'],
                            algorithm_type=row['algorithm_type'],
                            tokenizer_id=row['tokenizer_id'],
                            mapping=mapping_cql,  # unicode to string
                            mapping_name=mapping_name_cql,
                            status=row['status'],
                            train_test_size=str(train_test_size),
                            create_user=row['create_user'],
                            create_date=row['create_date']
                            )

            # cls.logger.debug('update_train_log_by_column >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
            cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def delete_train_log(cls, robot_id, model_id_list=None, modify_user=None):
        """
        刪除訓練紀錄 delete_train_log
        :param robot_id:
        :param model_id_list:
        :param modify_user:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()

            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id_list:
                cql_in = "'"
                for index, value in enumerate(model_id_list):
                    cql_in = cql_in + value.lower() + "'"
                    if index + 1 != len(model_id_list):
                        cql_in = cql_in + ", '"
                where_cql = where_cql + " and model_id in ({cql_in})".format(cql_in=cql_in)

        cql = "delete from {table}".format(table=cls.INTENT_TRAIN_LOG)
        cql = cql + where_cql + ";"
        # cls.logger.debug('delete_train_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_test_log(cls, robot_id, model_id=None):
        """
        查詢測試紀錄 query_test_log
        :param robot_id:
        :param model_id:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()

            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id:
                model_id = model_id.lower()
                where_cql = where_cql + " and model_id = '{model_id}'".format(model_id=model_id)

        cql = "select * from " + cls.INTENT_TEST_LOG + where_cql + ";"
        cls.logger.debug('query_test_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'accuracy_score': '',
                        'total_count': 0,
                        'correct_count': 0,
                        'f1_score': '',
                        'precision_score': '',
                        'recall_score': '',
                        'img': '',
                        'create_user': 'camp',
                        'create_date': '',
                        'error_evaluate_sentence': '{}'
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        pd_df = pd_df.sort_values(by=['create_date'], ascending=False).reset_index(drop=True)
        return pd_df

    @classmethod
    def insert_test_log(cls, robot_id, model_id, total_count, correct_count,
                        accuracy_score=-1, precision_score=-1, recall_score=-1, f1_score=-1,
                        img='',
                        modify_user='camp', error_evaluate_sentence='{}'):
        """
        插入測試紀錄 insert_test_log
        :param robot_id:
        :param model_id:
        :param total_count:
        :param correct_count:
        :param accuracy_score:
        :param precision_score:
        :param recall_score:
        :param f1_score:
        :param modify_user:
        :param img:
        :param error_evaluate_sentence:
        :return:
        """
        modify_user = modify_user.lower()
        # train_id = cls.get_uuid()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        # error_evaluate_sentence = ",".join(error_evaluate_sentence)

        # cql = ("insert into {table}(" +
        #        "robot_id, model_id, total_count, correct_count," +
        #        "accuracy_score, precision_score, recall_score, f1_score, " +
        #        "img, " +
        #        "create_user, create_date, error_evaluate_sentence)" +
        #        " values('{robot_id}', '{model_id}', {total_count}, {correct_count}, " +
        #        "'{accuracy_score}', '{precision_score}', '{recall_score}', '{f1_score}', " +
        #        "'{img}', " +
        #        "'{create_user}','{create_date}', '{error_evaluate_sentence}');")
        #
        # # cls.logger.debug(error_evaluate_sentence)
        # cql = cql.format(table=cls.INTENT_TEST_LOG,
        #                  robot_id=robot_id,
        #                  model_id=model_id,
        #                  total_count=total_count,
        #                  correct_count=correct_count,
        #                  accuracy_score=str(accuracy_score),
        #                  precision_score=str(precision_score),
        #                  recall_score=str(recall_score),
        #                  f1_score=str(f1_score),
        #                  img=img,
        #                  create_date=modify_date,
        #                  create_user=modify_user,
        #                  error_evaluate_sentence=error_evaluate_sentence
        #                  )
        #
        # cls.logger.debug('insert_test_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        # cls.dao.execCQL(cls.MSSQL_DB, cql)


        cql = ("insert into {table} values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        cql = cql.format(table=cls.INTENT_TEST_LOG)

        # 順序要跟mssql中的欄位順序一致
        param_tuple = (robot_id, model_id, str(accuracy_score), correct_count, modify_date,
                       modify_user, error_evaluate_sentence, str(f1_score), img, str(precision_score),
                       str(recall_score), total_count)

        cls.logger.debug('insert_test_log >>> : ' + cql + ' ; param_tuple >>> ' + str(param_tuple))
        # cls.exec_cql_transations_param_tuple([cql], [param_tuple])

        cls.dao.execCQLParamTuple(cls.MSSQL_DB, cql, param_tuple)



    @classmethod
    def update_test_log(cls, robot_id, model_id, modify_user, total_count, correct_count,
                        accuracy_score, precision_score, recall_score, f1_score,
                        img='',
                        error_evaluate_sentence='{}'):
        """
        更新測試紀錄 update_test_log
        :param robot_id:
        :param model_id:
        :param modify_user:
        :param total_count:
        :param correct_count:
        :param accuracy_score:
        :param precision_score:
        :param recall_score:
        :param f1_score:
        :param img:
        :param error_evaluate_sentence:
        :return:
        """
        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        # error_evaluate_sentence = ",".join(error_evaluate_sentence)

        cql = ("update {table}" +
               " set total_count={total_count}, correct_count={correct_count}, " +
               " accuracy_score='{accuracy_score}', precision_score='{precision_score}', " +
               " recall_score='{recall_score}', f1_score='{f1_score}', " +
               " img='{img}', " +
               " create_user='{create_user}', create_date='{create_date}', " +
               " error_evaluate_sentence=N'{error_evaluate_sentence}' " +
               " where robot_id='{robot_id}' and model_id='{model_id}'"
               ).format(table=cls.INTENT_TEST_LOG,
                        robot_id=robot_id,
                        model_id=model_id,
                        total_count=total_count,
                        correct_count=correct_count,
                        accuracy_score=str(accuracy_score),
                        precision_score=str(precision_score),
                        recall_score=str(recall_score),
                        f1_score=str(f1_score),
                        img=img,
                        create_date=modify_date,
                        create_user=modify_user,
                        error_evaluate_sentence=error_evaluate_sentence
                        )

        cls.logger.debug('update_test_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)


    @classmethod
    def update_test_log_by_column(cls, robot_id, model_id, modify_user=None,
                                  total_count=None, correct_count=None, accuracy_score=None,
                                  precision_score=None, recall_score=None, f1_score=None,
                                  img=None,
                                  error_evaluate_sentence=None):
        """
        更新測試紀錄 update_test_log_by_column
        :param robot_id:
        :param model_id:
        :param modify_user:
        :param total_count:
        :param correct_count:
        :param accuracy_score:
        :param precision_score:
        :param recall_score:
        :param f1_score:
        :param img:
        :param error_evaluate_sentence:
        :return:
        """

        modify_user = modify_user.lower()
        modify_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        columns_cql = ''
        if total_count:
            columns_cql = columns_cql + "total_count={total_count}, "

        if correct_count:
            columns_cql = columns_cql + "correct_count={correct_count}, "

        if accuracy_score:
            columns_cql = columns_cql + "accuracy_score='{accuracy_score}', "

        if precision_score:
            columns_cql = columns_cql + "precision_score='{precision_score}', "

        if recall_score:
            columns_cql = columns_cql + "recall_score='{recall_score}', "

        if f1_score:
            columns_cql = columns_cql + "f1_score='{f1_score}', "

        if img:
            columns_cql = columns_cql + "img='{img}', "

        if modify_user:
            columns_cql = columns_cql + "create_user='{create_user}', "

        if error_evaluate_sentence:
            columns_cql = columns_cql + "error_evaluate_sentence=N'{error_evaluate_sentence}', "

        cql = ("update {table}" +
               " set " + columns_cql +
               " create_date='{create_date}'"
               " where robot_id='{robot_id}' and model_id='{model_id}'"
               ).format(table=cls.INTENT_TEST_LOG,
                        robot_id=robot_id,
                        model_id=model_id,
                        total_count=total_count,
                        correct_count=correct_count,
                        accuracy_score=str(accuracy_score),
                        precision_score=str(precision_score),
                        recall_score=str(recall_score),
                        f1_score=str(f1_score),
                        img=img,
                        create_date=modify_date,
                        create_user=modify_user,
                        error_evaluate_sentence=error_evaluate_sentence
                        )

        cls.logger.debug('update_test_log_by_column >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def delete_test_log(cls, robot_id, model_id_list=None, modify_user=None):
        """
        刪除測試紀錄 delete_test_log
        :param robot_id:
        :param model_id_list:
        :param modify_user:
        :return:
        """
        where_cql = ""
        if robot_id:
            robot_id = robot_id.lower()

            where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)

            if model_id_list:
                cql_in = "'"
                for index, value in enumerate(model_id_list):
                    cql_in = cql_in + value.lower() + "'"
                    if index + 1 != len(model_id_list):
                        cql_in = cql_in + ", '"

                where_cql = where_cql + " and model_id in ({cql_in})".format(cql_in=cql_in)

        cql = "delete from {table}".format(table=cls.INTENT_TEST_LOG)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_test_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def remove_model_file_all_startwith(cls, robot_id, model_path_list=[], modify_user=None):
        """
        實體model刪除
        把robot_id開頭全部刪除 , 遞迴, 包含子資料夾
        :param robot_id:
        :param model_path_list:
        :param modify_user:
        :return:
        """
        cls.logger.debug("robot_id : " + robot_id)

        for model_path in model_path_list:
            cls.logger.debug("model_path : " + model_path)

            for root, dirs, files in os.walk(model_path):
                # cls.logger.debug root
                # cls.logger.debug dirs
                remove_list = []
                for name in files:
                    if name.startswith(robot_id):
                        # cls.logger.debug name
                        remove_list.append(name)

                        cls.logger.debug(remove_list)

                for item in remove_list:
                    os.remove(model_path + item)

    @classmethod
    def remove_model_file(cls, robot_id, model_path_list=[], sub_title_list=[], model_id_list=[], modify_user=None):
        """
        實體model刪除
        :param robot_id:
        :param model_path_list:
        :param sub_title_list:
        :param model_id_list:
        :param modify_user:
        :return:
        """
        if len(model_id_list) > 0:

            for model_id in model_id_list:
                for sub_title in sub_title_list:
                    for model_path in model_path_list:
                        from_model_name = robot_id + "+" + model_id + sub_title
                        cls.logger.debug("remove_path :{remove_path}".format(remove_path=model_path + from_model_name))
                        if os.path.isfile(model_path + from_model_name):
                            os.remove(model_path + from_model_name)

    # @classmethod
    # def remove_model_file(cls, robot_id, model_path, model_id_list=[], modify_user=None):
    #     """
    #     實體model刪除
    #     :param robot_id:
    #     :param model_path:
    #     :param model_id_list:
    #     :param modify_user:
    #     :return:
    #     """
    #     if len(model_id_list) > 0:
    #
    #         for model_id in model_id_list:
    #             from_model_name = robot_id + "+" + model_id + ".h5"
    #             from_tokenizer_name = robot_id + "+" + model_id + ".pickle"
    #             cls.logger.debug("remove_path :{remove_path}".format(remove_path=model_path + from_model_name))
    #             if os.path.isfile(model_path + from_model_name):
    #                 os.remove(model_path + from_model_name)
    #             if os.path.isfile(model_path + from_tokenizer_name):
    #                 os.remove(model_path + from_tokenizer_name)

    @classmethod
    def online_model_file(cls, robot_id, model_id, algorithm_type, modify_user=None):
        """
        實體model上線
        :param robot_id:
        :param model_id:
        :param algorithm_type:
        :param modify_user:
        :return:
        """
        # 將上線的model放在online folder
        path_online = Get_MyEnv().env_fs_model_path + "intent/online/"
        path_offline = Get_MyEnv().env_fs_model_path + "intent/"

        r_m_id = robot_id + "+" + model_id

        if algorithm_type == "DL":
            from_model_name = r_m_id + ".h5"
            # to_model_name = robot_id + ".h5"
            shutil.copy2(path_offline + from_model_name, path_online + from_model_name)

            from_tokenizer_name = r_m_id + ".pickle"
            # to_tokenizer_name = robot_id + ".pickle"
            shutil.copy2(path_offline + from_tokenizer_name, path_online + from_tokenizer_name)

            cls.logger.debug("from_copy_path :{from_copy_path} , to_copy_path :{to_copy_path}".format(
                from_copy_path=path_offline + from_model_name, to_copy_path=path_online + from_model_name))

        elif algorithm_type == "ML":
            from_model_name = r_m_id + ".joblib"
            # to_model_name = robot_id + ".joblib"
            shutil.copy2(path_offline + from_model_name, path_online + from_model_name)

            from_tokenizer_name = r_m_id + ".pickle"
            # to_tokenizer_name = robot_id + ".pickle"
            shutil.copy2(path_offline + from_tokenizer_name, path_online + from_tokenizer_name)

            cls.logger.debug("from_copy_path :{from_copy_path} , to_copy_path :{to_copy_path}".format(
                from_copy_path=path_offline + from_model_name, to_copy_path=path_online + from_model_name))

        elif algorithm_type == "BERT":
            from_model_name = r_m_id + ".bin"
            # to_model_name = robot_id + ".bin"
            shutil.copy2(path_offline + from_model_name, path_online + from_model_name)

            from_tokenizer_name = r_m_id + ".pickle"
            # to_tokenizer_name = robot_id + ".pickle"
            shutil.copy2(path_offline + from_tokenizer_name, path_online + from_tokenizer_name)

            cls.logger.debug("from_copy_path :{from_copy_path} , to_copy_path :{to_copy_path}".format(
                from_copy_path=path_offline + from_model_name, to_copy_path=path_online + from_model_name))

        else:
            cls.logger.error("There is no algorithm_type :{algorithm_type}".format(algorithm_type=algorithm_type))
            pass

    @classmethod
    def offline_model_file(cls, robot_id, algorithm_type, modify_user=None):
        """
        實體model下線, 因爲有dl or ml model副檔名不同, 避免有舊model沒刪乾淨,就採取該robot_id全部刪除
        :param robot_id:
        :param algorithm_type:
        :param modify_user:
        :return:
        """

        path_online = Get_MyEnv().env_fs_model_path + "intent/online/"

        from_dl_model_name = robot_id + ".h5"
        from_ml_model_name = robot_id + ".joblib"
        from_bert_model_name = robot_id + ".bin"
        from_tokenizer_name = robot_id + ".pickle"
        if os.path.isfile(path_online + from_dl_model_name):
            os.remove(path_online + from_dl_model_name)
        if os.path.isfile(path_online + from_ml_model_name):
            os.remove(path_online + from_ml_model_name)
        if os.path.isfile(path_online + from_bert_model_name):
            os.remove(path_online + from_bert_model_name)
        if os.path.isfile(path_online + from_tokenizer_name):
            os.remove(path_online + from_tokenizer_name)

        # if algorithm_type == "DL":
        #
        #     from_model_name = robot_id + ".h5"
        #     from_tokenizer_name = robot_id + ".pickle"
        #     if os.path.isfile(path_online + from_model_name):
        #         os.remove(path_online + from_model_name)
        #     if os.path.isfile(path_online + from_tokenizer_name):
        #         os.remove(path_online + from_tokenizer_name)
        #
        #     cls.logger.debug("remove_path :{remove_path}".format(remove_path=path_online + from_model_name))
        #
        # elif algorithm_type == "ML":
        #
        #     from_model_name = robot_id + ".joblib"
        #     from_tokenizer_name = robot_id + ".pickle"
        #
        #     if os.path.isfile(path_online + from_model_name):
        #         os.remove(path_online + from_model_name)
        #     if os.path.isfile(path_online + from_tokenizer_name):
        #         os.remove(path_online + from_tokenizer_name)
        #
        #     cls.logger.debug("remove_path :{remove_path}".format(remove_path=path_online + from_model_name))
        #
        # else:
        #     cls.logger.debug('pass')
        #     pass

    @classmethod
    def query_train_sentence_corpus(cls, skill_id, random=False, sentence_id=None, limit=None):
        """
        查詢訓練語料
        :param skill_id:
        :param random:
        :param sentence_id:
        :param limit:
        :return:
        """

        where_cql = ""
        if skill_id:
            skill_id = skill_id.lower()
            where_cql = where_cql + " where skill_id = '{skill_id}'".format(skill_id=skill_id)

            if sentence_id:
                where_cql = where_cql + " and sentence_id = {sentence_id}".format(sentence_id=sentence_id)

        # if limit:
        #     where_cql = where_cql + " limit {limit}".format(limit=limit)

        cql = "select * from " + cls.INTENT_TRAIN_SENTENCE_CORPUS + where_cql + ";"
        cls.logger.debug('query_train_sentence_corpus >>> : ' + cql)
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'sentence': ''
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        if random:
            pd_df = shuffle(pd_df)

        if limit:
            pd_df = pd_df[:limit]

        pd_df = pd_df.reset_index(drop=True)
        # pd_df = pd_df.sort_values(by=['skill_id'], ascending=True).reset_index(drop=True)
        return pd_df

    @classmethod
    def query_sentence_modify_log(cls, robot_id, start_date=None, end_date=None, sentence_set_id=None,
                                  modify_type=None, skill_id=None, sentence_id=None):
        """
        查詢sentence修改紀錄
        :param robot_id:
        :param start_date:
        :param end_date:
        :param sentence_set_id:
        :param modify_type:
        :param skill_id:
        :param sentence_id:
        :return:
        """

        robot_id = robot_id.lower()

        where_cql = " where robot_id = '" + robot_id + "' order by create_date_ymd desc"
        cql = "select top(100)* from " + cls.SENTENCE_MODIFY_LOG + where_cql + ";"
        cls.logger.debug('query_sentence_modify_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        top_pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)
        last_create_user = ''
        last_create_date = ''
        if len(top_pd_df) > 0:
            top_pd_df = top_pd_df.sort_values(by=['create_date'], ascending=[False]).reset_index(drop=True)

            # cls.logger.debug(top_pd_df)

            last_create_user = top_pd_df['create_user'][0]
            last_create_date = top_pd_df['create_date'][0]

            # cls.logger.debug('create_user')
            # cls.logger.debug(last_create_user)
            # cls.logger.debug(type(last_create_user))
            # cls.logger.debug(last_create_date)
            # cls.logger.debug(type(last_create_date))

        where_cql = ""
        where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)
        if start_date:
            start_date = start_date.lower()
            where_cql = where_cql + " and create_date_ymd >= '{start_date}'".format(start_date=start_date)

        if end_date:
            end_date = end_date.lower()
            where_cql = where_cql + " and create_date_ymd <= '{end_date}'".format(end_date=end_date)

        cql = "select * from " + cls.SENTENCE_MODIFY_LOG + where_cql + ";"
        cls.logger.debug('query_sentence_modify_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'sentence': '',
                        'create_user': 'camp',
                        'create_date': ''
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        if sentence_set_id:
            pd_df = pd_df[pd_df['sentence_set_id'] == sentence_set_id]

        if modify_type:
            pd_df = pd_df[pd_df['modify_type'] == modify_type]

        if skill_id:
            pd_df = pd_df[pd_df['skill_id'] == skill_id]

        if sentence_id:
            pd_df = pd_df[pd_df['sentence_id'] == sentence_id]

        pd_df = pd_df.sort_values(by=['create_date'], ascending=False).reset_index(drop=True)
        return pd_df, last_create_user, last_create_date

    @classmethod
    def insert_sentence_modify_log(cls, robot_id, sentence_set_id, modify_type,
                                   skill_id, sentence, create_user):
        """
        新增句子紀錄 insert_sentence_modify_log
        :param robot_id:
        :param sentence_set_id:
        :param modify_type:
        :param skill_id:
        :param sentence:
        :param create_user:
        :return:
        """

        create_user = create_user.lower()
        create_date_ymd = datetime.strftime(datetime.now(), cls.DATE_FORMAT_YMD)
        create_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)
        sentence_id = cls.get_uuid()

        cql = ("insert into {table}(" +
               "robot_id, create_date_ymd, sentence_set_id, modify_type, skill_id, sentence, sentence_id, " +
               " create_date, create_user)" +
               " values('{robot_id}', '{create_date_ymd}', '{sentence_set_id}', '{modify_type}', '{skill_id}', " +
               " N'{sentence}', '{sentence_id}', " +
               " '{create_date}', '{create_user}');")

        cql = cql.format(table=cls.SENTENCE_MODIFY_LOG,
                         robot_id=robot_id,
                         create_date_ymd=create_date_ymd,
                         sentence_set_id=sentence_set_id,
                         modify_type=modify_type,
                         skill_id=skill_id,
                         sentence_id=str(sentence_id),
                         sentence=sentence,
                         create_user=create_user,
                         create_date=create_date
                         )

        cls.logger.debug(
            'insert_sentence_modify_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def update_sentence_modify_log_by_column(cls, robot_id, sentence_set_id, modify_type,
                                             skill_id, sentence, sentence_id=None, create_user=None):
        """
        更新句子紀錄 update_sentence_modify_log_by_column
        :param robot_id:
        :param sentence_set_id:
        :param modify_type:
        :param skill_id:
        :param sentence:
        :param sentence_id:
        :param create_user:
        :return:
        """

        create_user = create_user.lower()
        create_date_ymd = datetime.strftime(datetime.now(), cls.DATE_FORMAT_YMD)
        create_date = datetime.strftime(datetime.now(), cls.DATE_FORMAT_NORMAL)

        columns_cql = ''

        if create_user:
            columns_cql = columns_cql + "create_user='{create_user}', "

        cql = ("update {table}" +
               " set " + columns_cql +
               " sentence=N'{sentence}', create_date='{create_date}'"
               " where robot_id='{robot_id}' and create_date_ymd='{create_date_ymd}' and " +
               " sentence_set_id='{sentence_set_id}' and modify_type='{modify_type}' and " +
               " skill_id='{skill_id}' and sentence_id='{sentence_id}'"
               ).format(table=cls.SENTENCE_MODIFY_LOG,
                        robot_id=robot_id,
                        create_date_ymd=create_date_ymd,
                        sentence_set_id=sentence_set_id,
                        modify_type=modify_type,
                        skill_id=skill_id,
                        sentence_id=str(sentence_id),
                        sentence=sentence,
                        create_user=create_user,
                        create_date=create_date
                        )

        cls.logger.debug(
            'update_sentence_modify_log_by_column >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def delete_sentence_modify_log(cls, robot_id, create_date_ymd=None, sentence_set_id=None, modify_type=None,
                                   skill_id=None, sentence_id_list=None, modify_user=None):
        """
        刪除更新句子紀錄
        :param robot_id:
        :param create_date_ymd:
        :param sentence_set_id:
        :param modify_type:
        :param skill_id:
        :param sentence_id_list:
        :param modify_user:
        :return:
        """

        where_cql = ""
        robot_id = robot_id.lower()
        where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)
        if create_date_ymd:
            create_date_ymd = create_date_ymd.lower()

            where_cql = where_cql + " and create_date_ymd = '{create_date_ymd}'".format(create_date_ymd=create_date_ymd)

            if sentence_set_id:
                sentence_set_id = sentence_set_id.lower()

                where_cql = where_cql + " and sentence_set_id = '{sentence_set_id}'".format(
                    sentence_set_id=sentence_set_id)

                if modify_type:
                    modify_type = modify_type.lower()
                    where_cql = where_cql + " and modify_type = '{modify_type}'".format(modify_type=modify_type)

                    if skill_id:
                        skill_id = skill_id.lower()
                        where_cql = where_cql + " and skill_id = '{skill_id}'".format(skill_id=skill_id)

                        if sentence_id_list:
                            cql_in = ""
                            for index, value in enumerate(sentence_id_list):
                                cql_in = cql_in + value.lower() + ""
                                if index + 1 != len(sentence_id_list):
                                    cql_in = cql_in + ", "

                            where_cql = where_cql + " and sentence_id in ({cql_in})".format(cql_in=cql_in)

        cql = "delete from {table}".format(table=cls.SENTENCE_MODIFY_LOG)
        cql = cql + where_cql + ";"
        cls.logger.debug('delete_test_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        cls.dao.execCQL(cls.MSSQL_DB, cql)

    @classmethod
    def query_semantic_model_log(cls, robot_id, start_date=None, end_date=None,
                                 start_confidence=None, end_confidence=None, predict_top_skill=None, caller=None):
        """
        查詢sentence修改紀錄
        :param robot_id:
        :param start_date:
        :param end_date:
        :param start_confidence:
        :param end_confidence:
        :param predict_top_skill:
        :param caller:
        :return:
        """

        where_cql = ""
        robot_id = robot_id.lower()
        where_cql = where_cql + " where robot_id = '{robot_id}'".format(robot_id=robot_id)
        if start_date:
            start_date = start_date.lower()
            where_cql = where_cql + " and predict_time_ymd >= '{start_date}'".format(start_date=start_date)

        if end_date:
            end_date = end_date.lower()
            where_cql = where_cql + " and predict_time_ymd <= '{end_date}'".format(end_date=end_date)

        cql = "select * from " + cls.SEMANTIC_MODEL_LOG + where_cql + ";"
        cls.logger.debug('query_sentence_modify_log >>> {} , keyspace >>> {}'.format(cql, cls.MSSQL_DB))
        pd_df = cls.dao.execCQLSelectToPandasDF(cls.MSSQL_DB, cql)

        # null補值, 以防新增欄位有null產生
        fill_na_dict = {'sentence': '',
                        'caller': 'camp',
                        'create_date': '',
                        'predict_time': ''
                        }

        pd_df = pd_df.fillna(value=fill_na_dict)

        # 是否信心值可以超過start_confidence
        # pd_df = pd_df[pd_df['is_predict_success'] == False]

        if start_confidence:
            pd_df = pd_df[pd_df['confidence_top'] >= str(start_confidence)]

        if end_confidence:
            pd_df = pd_df[pd_df['confidence_top'] <= str(end_confidence)]

        if predict_top_skill:
            pd_df = pd_df[pd_df['predict_top_skill'] == predict_top_skill]

        if caller:
            pd_df = pd_df[pd_df['caller'] == caller]

        pd_df = pd_df.sort_values(by=['predict_time_ymd'], ascending=False).reset_index(drop=True)
        # pd_df = pd_df.sort_values(by=['predict_time'], ascending=False).reset_index(drop=True)
        return pd_df
