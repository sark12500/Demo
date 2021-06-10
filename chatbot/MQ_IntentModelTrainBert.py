# encoding=UTF-8
# !flask/bin/python

import sys

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

import pika
import time, json, sys
from datetime import datetime
from MqDAO import MqDAO
import pandas as pd
import numpy as np
import logging

from MyEnv import Get_MyEnv
from Config_Helper import Get_HelperConfig
from Config_Format import Get_FormatConfig
from HelperIntentModelUtility_param import HelperIntentModelUtility
from Utility_Logger import UtilityLogger
from NLP_IntentModelFactory import *

from tensorflow import Graph, Session
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import scikitplot as skplt

if __name__ == '__main__':

    # log
    # 系統log只顯示error級別以上的
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    # 自訂log
    logger = logging.getLogger('MQ_IntentModelTrainBert.py')
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

    def train(ch, method, properties, body):

        logger.info(" [x] Received .")
        logger.info(body)

        body = json.loads(body)
        logger.info("mq callback time : " + str(datetime.now()))

        # body = {
        #     "robot_id": "hr.00001318",
        #     "model_id": "model_name",
        #     "sentence_set_id": "sentence_01",
        #     "algorithm": "textcnn",
        #     "modify_user": "charles.huang@quantatw.com",
        #     "train_test_split_size": 0.8,
        #     "sentence_max_len": 20
        # }

        robot_id = body['robot_id']
        model_id = body['model_id']
        sentence_set_id = body['sentence_set_id']
        algorithm = body['algorithm']
        modify_user = body['modify_user']

        # 向量長度 - 取最長的單字向量長度
        sentence_max_len = body.get('sentence_max_len', 10)
        # 訓練測試切割
        train_test_split_size = body.get('train_test_split_size', 0.8)

        # 接到queue的event status = GET_QUEUE
        HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                            model_id=model_id,
                                                            status="GET_QUEUE")

        sentence_df = HelperIntentModelUtility.query_train_sentence(robot_id=robot_id,
                                                                    sentence_set_id=sentence_set_id)

        train_sentence_skill_sum = sentence_df.groupby(["skill_id"]).size()

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
                if algorithm == "bert":
                    # BERT code
                    factory = IntenBertModelFactory()
                    model_param = {
                        "bert_base": "bert-base-chinese",
                        "batch_size": 16,
                        "epochs": 3,
                        "sentence_max_len": 20,
                        "no_decay": ['bias', 'gamma', 'beta'],
                        "attention_masks_column": 'attention_masks',
                        "optimizer_name": 'BertAdam',
                        "model_sub_name": ".bin",
                        "tokenizer_sub_name": ".pickle",
                        "bert_tokenizer_path": Get_MyEnv().env_fs_model_path + "bert/bert-base-chinese-vocab.txt",
                        "bert_model": Get_MyEnv().env_fs_model_path + "bert/bert-base-chinese.tar.gz"
                    }

                else:
                    # 此區塊不該發生
                    model_param = {}
                    factory = None
                    error_msg = "algorithm : {algorithm} has not implemented !!".format(algorithm=algorithm)
                    logger.error(error_msg)
                    # data = [request.json]
                    # return jsonify(code=StateCode_HelperData.ModelNotExist,
                    #                message=error_msg,
                    #                data=data
                    #                )

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

                logger.debug('nfs_model_id : {}'.format(nfs_model_id))
                logger.debug('nfs_tokenizer_id : {}'.format(nfs_tokenizer_id))

                HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                                    model_id=model_id,
                                                                    algorithm_param=algorithm_param_json,
                                                                    algorithm_type=model.algorithm_type,
                                                                    tokenizer_id=feature_transformer_name,
                                                                    mapping=model.mapping,
                                                                    mapping_name=model.mapping_name,
                                                                    status="TRAIN",
                                                                    nfs_model_id=nfs_model_id,
                                                                    nfs_tokenizer_id=nfs_tokenizer_id)

                try:

                    logger.debug('============ train_model_start ============')

                    """
                    訓練模型
                    """
                    model.train_model(x_train,
                                      y_train)

                    """
                    model evaluate
                    直接把訓練句丟model預測, 切出驗證資料會使訓練資料過少(TODO: 如果能有夠多句子或自動生成句子就可以切)
                    並把結果計算統計值 EX : accuracy, f1-score 用來參考
                    """
                    HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                                        model_id=model_id,
                                                                        status="EVALUATE")

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
                    log_id = "MQ_IntentModelTrainBert"
                    utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

                    # return jsonify(code=StateCode.Unexpected,
                    #                data=[],
                    #                message=msg
                    #                ), 999

                try:
                    """
                    持久化實體檔案
                    1. 模型
                    2. 評估 - ROC圖
                    3. 評估 - confusion_matrix圖
                    4. 評估 - PRC圖
                    5. 模型 - 訓練過程
                    """
                    HelperIntentModelUtility.update_train_log_by_column(robot_id=robot_id,
                                                                        model_id=model_id,
                                                                        status="SAVE")

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
                    log_id = "MQ_IntentModelTrainBert"
                    utility_logger.save_log(MSSQL_DB, HELPER_ERROR_LOG_TABLE, log_id, msg)

                    # return jsonify(code=StateCode.Unexpected,
                    #                data=[],
                    #                message=msg
                    #                ), 999

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

                # data = dict(
                #     accuracy_score=acc_score,
                #     precision_score=recall_score,
                #     recall_score=recall_score,
                #     f1_score=f1_score,
                #     error_evaluate_sentence=error_evaluate_list
                # )

                # return jsonify(code=StateCode.Success,
                #                message="success",
                #                data=[data]
                #                )


    # queue啓動
    queue_id = Get_MyEnv().env_mq_name_intent_train_bert_queue
    mq.start_consuming(queue_id, callback=train)
