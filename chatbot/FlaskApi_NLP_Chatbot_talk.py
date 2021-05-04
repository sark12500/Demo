#!flask/bin/python
# -*- coding: utf-8 -*-
import sys
import os, inspect

# reload(sys)
# sys.setdefaultencoding('utf-8')

from flask_cors import CORS
import pandas as pd
import logging
import json

import requests, traceback
import re
from datetime import datetime
# import asyncio
import threading
import time
from flask import Flask, jsonify
from flask_restful import fields, reqparse

from FlaskApi_CheckAttribute import *
from MyEnv import Get_MyEnv
from MssqlDAO import MssqlDAO
# from Utility_Logger import UtilityLogger
from StateCode import StateCode
from Config_Format import Get_FormatConfig
from FlaskApi_CheckAttribute import *
from NLP_JiebaSegmentor import JiebaSegmentor, Get_JiebaSegmentor
from NLP_ChatBot import *

# from NLP_WikiService import WikiPedia
# from NLP_GoogleService import GoogleSearch
from NLP_TfidfGensim import TfidfGensim
import emoji

from flask_restplus import Api, Resource, fields, Namespace
from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app,
          version="2020.12.28",
          title="聊天機器人服務API文件 - talk",
          description="")
CORS(app)

nlp_ns = api.namespace("api/chatbot", description="聊天機器人服務 - talk(點擊展開)")

free_talk_input_model = api.model(
    'free_talk_input_model', {
        'input_text': fields.String(required=True,
                                    description='輸入文字',
                                    example='桃園哪裏有好吃的食物'),
        'chatbot_id': fields.String(required=True,
                                    description='回答特定領域的機器人',
                                    example='food'),
        'top_n_article': fields.Integer(required=True,
                                        description='文章對應數目',
                                        example=10),
        'top_n_comment': fields.Integer(required=True,
                                        description='回答句子的顯示數目',
                                        example=20),
        'sort_by': fields.String(required=True,
                                 description='回答句子排序方式(random:隨機; comment_like_count:讚數, comment_created_at:發表時間, article_like_count:文章讚數, article_comment_count:文章回應數)',
                                 example='random'),

    }
)

@nlp_ns.route('/freeTalk')
class FreeTalk(Resource):

    @nlp_ns.expect(free_talk_input_model)
    # @requires_free_talk_parameters
    def post(self):
        """
        閒聊
        input
        {
            "input_text":"臺南美食推薦",
            "chatbot_id":"food",
            "top_n_article":10,
            "top_n_comment":20,
            "sort_by": "random"
        }
        """

        parser = reqparse.RequestParser()
        parser.add_argument('input_text', required=True)
        parser.add_argument('chatbot_id', required=True)
        parser.add_argument('top_n_article', type=int, required=True)
        parser.add_argument('top_n_comment', type=int, required=True)
        parser.add_argument('sort_by', required=True)
        args = parser.parse_args()

        logger.debug("DB : {}".format(Get_MyEnv().env_mssql_db))

        param_input_text = args.get('input_text', None)
        # param_chatbot_id_list = request.json.get('chatbot_id_list', [])
        # param_chatbot_id_list = [x.lower() for x in param_chatbot_id_list]
        param_chatbot_id = args.get('chatbot_id', None)
        if param_chatbot_id:
            param_chatbot_id = param_chatbot_id.lower()
        else:
            return jsonify(code=StateCode.Unexpected,
                           data=[],
                           message="chatbot_id is None"
                           ), 999

        param_top_n_article = args.get('top_n_article', 10)
        param_top_n_comment = args.get('top_n_comment', 20)
        param_sort_by = args.get('sort_by', 'random')

        if param_input_text is None or param_input_text == '':
            return jsonify(code=StateCode.Unexpected,
                           message="input_text is empty",
                           data=[]
                           )

        # if param_chatbot_id_list != "*":
        #     param_chatbot_id_list = [x.lower() for x in param_chatbot_id_list]

        param_sort_by = param_sort_by.lower()
        sort_by_type = ['random', 'comment_like_count', 'comment_created_at',
                        'article_like_count', 'article_comment_count']
        if param_sort_by not in sort_by_type:
            param_sort_by = 'random'

        chatbot = FreeTalkChatBot(tokenizer=jieba, tfidf_model=tfidf_gensim)

        try:
            # 預測答案
            theme = theme_factory.create(param_chatbot_id)
            if theme:
                ans_df, msg = chatbot.talk(theme=theme,
                                           input_text=param_input_text,
                                           top_n_article=param_top_n_article,
                                           top_n_comment=param_top_n_comment,
                                           sort_by=param_sort_by)
            else:
                return jsonify(code=StateCode.Unexpected,
                               data=[],
                               message="There is no chatbot that chatbot_id is {}".format(param_chatbot_id)
                               )

        except Exception as e:
            #    logger.debug(e)
            error_class = e.__class__.__name__  # 取得錯誤類型
            detail = e.args[0]  # 取得詳細內容
            cl, exc, tb = sys.exc_info()  # 取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
            fileName = lastCallStack[0]  # 取得發生的檔案名稱
            lineNum = lastCallStack[1]  # 取得發生的行號
            funcName = lastCallStack[2]  # 取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            logger.debug(errMsg)

            return jsonify(code=StateCode.Unexpected,
                           message=errMsg,
                           data=[]
                           )

        # result
        # result['comment_content'][0]

        data = []

        if ans_df is not None:
            for index, row in ans_df.iterrows():
                result = dict(
                    article_information="https://www.dcard.tw/service/api/v2/posts/" + str(row['article_id']),
                    article_title=row['article_title'],
                    article_topics=row['article_topics'],
                    comment_content=row['comment_content'],
                )
                data.append(result)

        return jsonify(code=StateCode.Success,
                       message=msg,
                       data=data
                       )

        # except Exception as e:
        #
        #     # utility_logger = UtilityLogger()
        #     # msg = utility_logger.except_error_msg(sys.exc_info())
        #     # logger.error(msg)
        #     # log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
        #     # utility_logger.save_log(HELPER_KEYSPACE, HELPER_ERROR_LOG_TABLE, log_id, msg)
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    data=[],
        #                    message="msg"
        #                    ), 999


@nlp_ns.route('/freeTalkSkillApi')
class FreeTalkSkillApi(Resource):

    # @nlp_ns.expect(free_talk_input_model)
    # @requires_free_talk_parameters
    def post(self):
        """
        閒聊 skill api
        input
        {
            "version_id": "1.1",
            "sequence": "owen-lin@quantatw.com_2020-12-22 15:18:58.032",
            "skill_id": "faq19999",
            "robot_id": "skywalker.00001318-owen-lin@quantatw.com",
            "user": {
                "camp_id": "owen-lin@quantatw.com",
                "attributes": [
                    {
                        "key": "employee_cht_name",
                        "value": "林慶文"
                    },
                    {
                        "key": "employee_eng_name",
                        "value": "Owen Lin"
                    },
                    {
                        "key": "ext_number",
                        "value": "19921"
                    },
                    {
                        "key": "company_id",
                        "value": "00001318"
                    },
                    {
                        "key": "email",
                        "value": "Owen-Lin@quantatw.com"
                    }
                ]
            },
            "device": {
                "device_id": "web_20201222151858033",
                "device_type": "W",
                "attributes": [
                    {
                        "key": "session_id",
                        "value": "af02d01f-05c4-4a92-999e-2bf0b22c7075"
                    },
                    {
                        "key": "env",
                        "value": "2"
                    }
                ],
                "trigger": "H",
                "mailbox_mail_id": "",
                "mailbox_mail_attach": [

                ],
                "mailbox_mail_guid": "",
                "rpa_agent_id": ""
            },
            "input_text": "我的薪資單打不開",
            "intent": "19999常見問題諮詢",
            "skill_json": {
            },
            "slot": [
                {
                    "key": "question",
                    "value": "我的薪資單打不開",
                    "item_id": "我的薪資單打不開",
                    "json": {
                    }
                }
            ],
            "missing_parameter_info": [

            ],
            "is_confirmed": false,
            "event_id": "2475083"
        }
        """

        param_input_text = request.json.get('input_text', None)

        # logger.debug("DB : {}".format(Get_MyEnv().env_mssql_db))

        param_chatbot_id = request.json.get('skill_id', None)
        if param_chatbot_id:
            param_chatbot_id = param_chatbot_id.lower()
        else:
            logger.debug("chatbot_id is None")
            # return jsonify(code=StateCode.Unexpected,
            #                data=[],
            #                message="chatbot_id is None"
            #                )

        param_top_n_article = 10
        param_top_n_comment = 20
        param_sort_by = 'random'

        if param_input_text is None or param_input_text == '':
            logger.debug("input_text is empty")
            # return jsonify(code=StateCode.Unexpected,
            #                message="input_text is empty",
            #                data=[]
            #                )

        chatbot = FreeTalkChatBot(tokenizer=jieba, tfidf_model=tfidf_gensim)

        # try:

        # 預測答案
        ans_df = pd.DataFrame({})
        theme = theme_factory.create(param_chatbot_id)
        if theme:
            ans_df, msg = chatbot.talk(theme=theme,
                                       input_text=param_input_text,
                                       top_n_article=param_top_n_article,
                                       top_n_comment=param_top_n_comment,
                                       sort_by=param_sort_by)
        else:
            logger.debug("There is no chatbot that chatbot_id is {}".format(param_chatbot_id))
            # return jsonify(code=StateCode.Unexpected,
            #                data=[],
            #                message="There is no chatbot that chatbot_id is {}".format(param_chatbot_id)
            #                )

        # except Exception as e:
        #     #    logger.debug(e)
        #     error_class = e.__class__.__name__  # 取得錯誤類型
        #     detail = e.args[0]  # 取得詳細內容
        #     cl, exc, tb = sys.exc_info()  # 取得Call Stack
        #     lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        #     fileName = lastCallStack[0]  # 取得發生的檔案名稱
        #     lineNum = lastCallStack[1]  # 取得發生的行號
        #     funcName = lastCallStack[2]  # 取得發生的函數名稱
        #     errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        #     logger.debug(errMsg)
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    message=errMsg,
        #                    data=[]
        #                    )
        if ans_df is not None and len(ans_df) > 0:
            answer = ans_df['comment_content'][0]
        else:
            answer = "我還在學習中..."

        logger.debug("answer : {}".format(answer))

        # save log


        result = {
            "code": "0",
            "message": "",
            "data": {
                "is_end": True,
                "need_confirm": False,
                "can_modify": False,
                "modify_slots": [

                ],
                "saas_event_id": "chatbot_charles_test01",
                "directive_items": [

                ],
                "skill_json": "",
                "push_to_display": {
                    "card": {
                        "type": "1",
                        "notify_type": "0",
                        "wait_user_response": False,
                        "action": "1",
                        "info": [
                            {
                                "title": answer,
                                "text": answer,
                                "rich_contents": [

                                ],
                                "chatroom_text": answer,
                                "item_list": [

                                ],
                                "recommend_text": "",
                                "mail_attach": [

                                ]
                            }
                        ]
                    }
                },
                "device": {
                    "device_id": "",
                    "device_type": "S",
                    "attributes": [

                    ],
                    "trigger": "H",
                    "mailbox_mail_id": "",
                    "mailbox_mail_attach": [

                    ],
                    "mailbox_mail_guid": "",
                    "rpa_agent_id": ""
                }
            }
        }

        #
        # # if param_chatbot_id_list != "*":
        # #     param_chatbot_id_list = [x.lower() for x in param_chatbot_id_list]
        #
        # param_sort_by = param_sort_by.lower()
        # sort_by_type = ['random', 'comment_like_count', 'comment_created_at',
        #                 'article_like_count', 'article_comment_count']
        # if param_sort_by not in sort_by_type:
        #     param_sort_by = 'random'



        # chatbot = FreeTalkChatBot(tokenizer=jieba, tfidf_model=tfidf_gensim)
        #
        # # try:
        # # 預測答案
        # theme = theme_factory.create(param_chatbot_id)
        # if theme:
        #     ans_df, msg = chatbot.talk(theme=theme,
        #                                input_text=param_input_text,
        #                                top_n_article=param_top_n_article,
        #                                top_n_comment=param_top_n_comment,
        #                                sort_by=param_sort_by)
        # else:
        #     return jsonify(code=StateCode.Unexpected,
        #                    data=[],
        #                    message="There is no chatbot that chatbot_id is {}".format(param_chatbot_id)
        #                    )

        # except Exception as e:
        #     #    logger.debug(e)
        #     error_class = e.__class__.__name__  # 取得錯誤類型
        #     detail = e.args[0]  # 取得詳細內容
        #     cl, exc, tb = sys.exc_info()  # 取得Call Stack
        #     lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        #     fileName = lastCallStack[0]  # 取得發生的檔案名稱
        #     lineNum = lastCallStack[1]  # 取得發生的行號
        #     funcName = lastCallStack[2]  # 取得發生的函數名稱
        #     errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        #     logger.debug(errMsg)
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    message=errMsg,
        #                    data=[]
        #                    )

        # result
        # result['comment_content'][0]

        data = []

        # if ans_df is not None:
        #     for index, row in ans_df.iterrows():
        #         result = dict(
        #             article_information="https://www.dcard.tw/service/api/v2/posts/" + str(row['article_id']),
        #             article_title=row['article_title'],
        #             article_topics=row['article_topics'],
        #             comment_content=row['comment_content'],
        #         )
        #         data.append(result)

        return jsonify(result)

        # except Exception as e:
        #
        #     # utility_logger = UtilityLogger()
        #     # msg = utility_logger.except_error_msg(sys.exc_info())
        #     # logger.error(msg)
        #     # log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
        #     # utility_logger.save_log(HELPER_KEYSPACE, HELPER_ERROR_LOG_TABLE, log_id, msg)
        #
        #     return jsonify(code=StateCode.Unexpected,
        #                    data=[],
        #                    message="msg"
        #                    ), 999



#
# @nlp_ns.route('/test')
# class test(Resource):
#
#     def post(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('name', required=True)
#         parser.add_argument('position', required=True)
#         parser.add_argument('addr', type=int, action='append')
#         parser.add_argument('min_wage', type=int)
#         parser.add_argument('max_wage', type=int)
#         parser.add_argument('description', required=True)
#
#         args = parser.parse_args()
#         logger.debug(args)
#
#         return jsonify(code=StateCode.Success,
#                        data=[],
#                        message=""
#                        )

if __name__ == "__main__":

    # log
    # 系統log只顯示error級別以上的
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    # 自訂log
    logger = logging.getLogger('FlaskAPi_NLP_Chatbot.py')
    logger.setLevel(logging.DEBUG)

    HELPER_KEYSPACE = Get_MyEnv().env_helper_keyspace
    NLP_KEYSPACE = Get_MyEnv().env_nlp_keyspace

    config = Get_FormatConfig()
    DATE_FORMAT_NORMAL = config.DATE_FORMAT_NORMAL

    # init jieba segmentor
    jieba = Get_JiebaSegmentor()
    # # init hanlp segmentor
    # segment = Get_HanlpSegmentor()

    tfidf_model_path = Get_MyEnv().env_fs_model_path + "tfidf/tfidf.model"
    tfidf_dictionary_path = Get_MyEnv().env_fs_model_path + "tfidf/tfidf_corpus_dict"
    stopwords_path = Get_MyEnv().env_fs_jieba_path + "stopwords.txt"
    tfidf_gensim = TfidfGensim(model_path=tfidf_model_path, dictionary_path=tfidf_dictionary_path,
                               stopword_path=stopwords_path, segmentor=jieba)

    theme_factory = ThemeFactory()

    # w2v_model_path = Get_MyEnv().env_fs_model_path + "word2vec/word2vec_tw.model"
    # w2v_gensim = WordToVecGensim(w2v_model_path)

    # # dcard 髒話
    # dirtywords_path = Get_MyEnv().env_fs_jieba_path + "dcard_dirtywords.txt"
    # dirtyword_set = set()
    # with open(dirtywords_path, 'r') as dirtywords:
    #     for dirtyword in dirtywords:
    #         dirtyword_set.add(dirtyword.strip('\n').decode('utf-8'))
    #
    # dirty_word_rule = ''
    # for dw in dirtyword_set:
    #     dirty_word_rule = dirty_word_rule + dw + '|'
    #
    # dirty_word_rule = '(' + dirty_word_rule[:len(dirty_word_rule) - 1] + ')'

    dao = MssqlDAO(ip=Get_MyEnv().env_mssql_ip,
                   account=Get_MyEnv().env_mssql_account,
                   password=Get_MyEnv().env_mssql_password)

    logger.info('api run !!')

    if Get_MyEnv().env_oo == Get_MyEnv().LOCAL:
        app.run(port=5022, host='192.168.95.27', debug=False, use_reloader=False, threaded=True)
    else:
        app.run(port=5022, host='0.0.0.0', debug=True, use_reloader=False, threaded=True)

## no use
#
# @nlp_ns.route('/freeTalk')
# class FreeTalk(Resource):
#
#     @nlp_ns.expect(free_talk_input_model)
#     @requires_free_talk_parameters
#     def post(self):
#         """
#         閒聊
#         input
#         {
#             "input_text":"臺南美食推薦",
#             "chatbot_id":"food",
#             "top_n_article":10,
#             "top_n_comment":20,
#             "sort_by": "random"
#         }
#         """
#
#         logger.debug("DB : {}".format(Get_MyEnv().env_mssql_db))
#
#         param_input_text = request.json.get('input_text', None)
#         # param_chatbot_id_list = request.json.get('chatbot_id_list', [])
#         # param_chatbot_id_list = [x.lower() for x in param_chatbot_id_list]
#         param_chatbot_id = request.json.get('chatbot_id', None)
#         if param_chatbot_id:
#             param_chatbot_id = param_chatbot_id.lower()
#         else:
#             return jsonify(code=StateCode.Unexpected,
#                            data=[],
#                            message="chatbot_id is None"
#                            ), 999
#
#         param_top_n_article = request.json.get('top_n_article', 10)
#         param_top_n_comment = request.json.get('top_n_comment', 20)
#         param_sort_by = request.json.get('sort_by', 'random')
#
#         if param_input_text is None or param_input_text == '':
#             return jsonify(code=StateCode.Unexpected,
#                            message="input_text is empty",
#                            data=[]
#                            )
#
#         # if param_chatbot_id_list != "*":
#         #     param_chatbot_id_list = [x.lower() for x in param_chatbot_id_list]
#
#         param_sort_by = param_sort_by.lower()
#         sort_by_type = ['random', 'comment_like_count', 'comment_created_at',
#                         'article_like_count', 'article_comment_count']
#         if param_sort_by not in sort_by_type:
#             param_sort_by = 'random'
#
#         dcard_query_table_chatbot = "dcard_query_table_" + param_chatbot_id + "_chatbot"
#         dcard_word_article_chatbot = "dcard_word_article_" + param_chatbot_id + "_chatbot"
#
#         # category_word_article_dic = {"food": dcard_word_article_food_chatbot}
#         # category_query_table_dic = {"food": dcard_query_table_food_chatbot}
#
#         def find_keyword(text):
#
#             # 依照TFIDF給權重取關鍵字
#             text = text.lower()
#
#             words_df = jieba.pseg_lcut_combie_num_eng(text)
#             cut_len = len(words_df)
#
#             tfidf_words_df = tfidf_gensim.predict(text,
#                                                   min_tfidf=0)
#
#             # if word is not in tfidf model, word will be removed. So I need to compare with jieba cut legth.
#             if len(tfidf_words_df) == cut_len:
#
#                 # 增加index欄位當權重用
#                 words_df = tfidf_words_df.reset_index()
#                 word_len = len(words_df)
#
#                 def set_weight_udf(row):
#                     return word_len - row['index']
#
#                 words_df['weight'] = words_df.apply(set_weight_udf, axis=1)
#
#             else:
#
#                 # 依照詞性取關鍵字
#                 # words_df = jieba.pseg_lcut_combie_num_eng(text)
#
#                 v_tag = ['v', 'vd', 'vg', 'vi', 'vn', 'vq', 'vt']
#                 n_tag = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']
#                 a_tag = ['a', 'ad', 'ag', 'an']
#                 m_tag = ['m', 'eng', 'm+eng']
#
#                 tags = v_tag + n_tag + a_tag + m_tag
#                 words_df = words_df[words_df['sp'].isin(tags)]
#                 words_df = words_df.rename(columns={"word": "keyword"})
#                 words_df['tfidf'] = 1
#                 words_df['weight'] = 1
#
#             logger.debug(words_df)
#
#             return words_df
#
#         def query_word_article(chatbot_id, keyword_df):
#             """
#             取得關鍵字對應的文章
#             """
#
#             word_list_str = "'" + "','".join(list(keyword_df['keyword'])) + "'"
#
#             article_df = pd.DataFrame({})
#             # by參數查詢版, * 表示所有版都查詢
#             # # if category == ["*"]:
#             # if chatbot_id == "*":
#             #     pass
#             #
#             # else:
#
#             # for cc in category:
#             word_article_table = dcard_word_article_chatbot
#             if word_article_table:
#                 sql = ("select article_id, article_title, article_topics, article_like_count, " +
#                        " article_comment_count, article_created_at,  article_updated_at , " +
#                        "count(article_id) as count from " +
#                        word_article_table +
#                        " where keyword in (" + word_list_str + ") group by article_id , " +
#                        " article_title, article_topics, article_like_count," +
#                        " article_like_count,  article_comment_count, article_created_at, " +
#                        " article_updated_at order by count desc ")
#                 #             logger.debug(sql)
#                 single_df = dao.execCQLSelectToPandasDF(Get_MyEnv().env_mssql_db, sql)
#                 article_df = article_df.append(single_df)
#
#             logger.debug(article_df)
#
#             return article_df
#
#         def article_similarly_calculate(df, by_column):
#             """
#             取得最大關鍵字出現數量的文章, 表示與input_text最像
#             """
#             df = df.sort_values(by=[by_column], ascending=False).reset_index(drop=True)
#
#             max_count = df[by_column][0]
#             logger.debug("max_{} : {}".format(by_column, max_count))
#
#             df = df[df[by_column] == max_count]
#             logger.debug('len : {}'.format(len(df)))
#
#             return df.reset_index(drop=True)
#
#         def article_filter_by_tfidf(df, keyword_df):
#             """
#             關鍵字match數量相同時, 依照tfidf weight排序
#             """
#
#             # 輸入文字的單字與weight組成的字典
#             word_weight_dict = dict(zip(keyword_df.keyword, keyword_df.weight))
#             keyword_list = list(keyword_df['keyword'])
#
#             def calculate_weight_udf(row):
#
#                 weight = 0
#                 for ww in keyword_list:
#                     if ww in row['article_title'] or ww in row['article_topics']:
#                         weight = weight + word_weight_dict[ww]
#
#                 return weight
#
#             df['weight'] = df.apply(calculate_weight_udf, axis=1)
#
#             return df.sort_values(by=['weight'], ascending=False)
#
#         def query_comment(chatbot_id, df, top_n=None):
#             """
#             取得回文
#             """
#
#             if top_n and top_n < 1:
#                 top_n = 1
#
#             #     logger.debug(top_n)
#
#             article_id_list = [str(x) for x in list(df['article_id'])]
#             article_id_list_str = "'" + "','".join(article_id_list) + "'"
#
#             comment_df = pd.DataFrame({})
#             # for cc in category:
#
#             if top_n:
#                 sql = ("select top(" + str(top_n) + ")* from " + dcard_query_table_chatbot +
#                        " where article_id in (" + article_id_list_str + ") order by comment_like_count desc")
#             else:
#
#                 sql = ("select * from " + dcard_query_table_chatbot +
#                        " where article_id in (" + article_id_list_str + ") order by comment_like_count desc")
#             #         logger.debug(sql)
#
#             single_df = dao.execCQLSelectToPandasDF(Get_MyEnv().env_mssql_db, sql)
#             comment_df = comment_df.append(single_df)
#
#             return comment_df
#
#         def predict(input_text, chatbot_id, top_n_article=10, top_n_comment=20, sort_by='random'):
#
#             # if len(category) == 0:
#             #     return None, '......沒有對應的版'
#
#             # 找出input_text關鍵字
#             input_text = input_text.lower()
#
#             keyword_df = find_keyword(input_text)
#
#             logger.debug(keyword_df)
#
#             # 取出input_text關鍵字與相關 的文章title
#             article_df = query_word_article(chatbot_id, keyword_df)
#
#             if len(article_df) == 0:
#                 return None, '......沒有關鍵字對應的文章'
#
#             # 取得最大關鍵字出現數量的文章, 表示與input_text最像
#             article_df = article_similarly_calculate(article_df, 'count')
#
#             # 關鍵字match數量相同時, 依照tfidf weight排序
#             article_df = article_filter_by_tfidf(article_df, keyword_df)
#
#             # logger.debug(article_df.head())
#
#             # 取得最大weight, 表示與input_text最像
#             article_df = article_similarly_calculate(article_df, 'weight')
#
#             logger.debug("len(print(article_df)) : {}".format(len(article_df)))
#
#             # 文章過多時, 近期受歡迎文章優先取出top n
#             if len(article_df) > top_n_article:
#                 article_df = article_df.sort_values(by=['article_comment_count', 'article_created_at'],
#                                                     ascending=False).reset_index(drop=True)[:top_n_article]
#
#             logger.debug(article_df)
#
#             # 取得回文
#             comment_df = query_comment(chatbot_id, article_df)
#
#             if len(comment_df) == 0:
#                 return None, '......沒有回文'
#
#             #     comment_df = comment_df[comment_df.comment_content.str.contains('原po') == False]
#             #     comment_df = comment_df[comment_df.article_topics.str.contains('食記') == False]
#
#             if sort_by == "random":
#                 comment_df = comment_df.sample(frac=1).reset_index(drop=True)
#             else:
#                 comment_df = comment_df.sort_values(by=[sort_by],
#                                                     ascending=False).reset_index(drop=True)
#
#             comment_df = comment_df[:top_n_comment]
#             comment_df = comment_df[
#                 ['article_id', 'article_title', 'article_topics',
#                  'comment_content', 'comment_like_count', 'comment_created_at']]
#
#             return comment_df, 'success'
#
#         try:
#             # 預測答案
#             ans_df, msg = predict(input_text=param_input_text, chatbot_id=param_chatbot_id,
#                                   top_n_article=param_top_n_article, top_n_comment=param_top_n_comment,
#                                   sort_by=param_sort_by)
#
#         except Exception as e:
#             #    logger.debug(e)
#             error_class = e.__class__.__name__  # 取得錯誤類型
#             detail = e.args[0]  # 取得詳細內容
#             cl, exc, tb = sys.exc_info()  # 取得Call Stack
#             lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
#             fileName = lastCallStack[0]  # 取得發生的檔案名稱
#             lineNum = lastCallStack[1]  # 取得發生的行號
#             funcName = lastCallStack[2]  # 取得發生的函數名稱
#             errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
#             logger.debug(errMsg)
#
#             return jsonify(code=StateCode.Unexpected,
#                            message=errMsg,
#                            data=[]
#                            )
#
#         # result
#         # result['comment_content'][0]
#
#         data = []
#
#         if ans_df is not None:
#             for index, row in ans_df.iterrows():
#                 result = dict(
#                     article_information="https://www.dcard.tw/service/api/v2/posts/" + str(row['article_id']),
#                     article_title=row['article_title'],
#                     article_topics=row['article_topics'],
#                     comment_content=row['comment_content'],
#                 )
#                 data.append(result)
#
#         return jsonify(code=StateCode.Success,
#                        message=msg,
#                        data=data
#                        )
#
#         # except Exception as e:
#         #
#         #     # utility_logger = UtilityLogger()
#         #     # msg = utility_logger.except_error_msg(sys.exc_info())
#         #     # logger.error(msg)
#         #     # log_id = self.__class__.__name__ + "_" + inspect.getframeinfo(inspect.currentframe()).function
#         #     # utility_logger.save_log(HELPER_KEYSPACE, HELPER_ERROR_LOG_TABLE, log_id, msg)
#         #
#         #     return jsonify(code=StateCode.Unexpected,
#         #                    data=[],
#         #                    message="msg"
#         #                    ), 999