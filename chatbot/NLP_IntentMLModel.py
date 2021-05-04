# encoding=UTF-8
# !flask/bin/python


from abc import abstractmethod, ABCMeta

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB, GaussianNB
# from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from scipy.stats import randint as sp_randint, uniform, expon
from joblib import dump, load
import numpy as np
import pandas as pd
import pickle

from MyEnv import Get_MyEnv
from NLP_IntentModel import IntentModel
from NLP_IntentPreprocessing import IntentPreprocessing
from NLP_JiebaSegmentor import Get_JiebaSegmentor


class IntentMLModel(IntentModel):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(IntentMLModel, self).__init__()
        self.algorithm_type = "ML"
    @abstractmethod
    def build_model(self):
        pass

    def save_log(self, _path):
        """
        存成實體log.txt
        :param _path:
        :return:
        """

        pass

    def summary(self):

        self.logger.debug(self.model.summary())

    def get_model(self):

        if self.model:
            return self.model
        else:
            self.logger.warning('please train model first !!')

    def get_model_param(self):

        if self.model_param:
            return self.model_param
        else:
            self.logger.warning('please train model first !!')

    def preprocessing(self, sentence_df, input_column, output_column):
        """
        資料前處理
        :param sentence_df:
        :param input_column:
        :param output_column:
        :return:
        """
        # 資料打散
        # sentence_df = IntentPreprocessing.shuffle(sentence_df)

        # 去除空白
        sentence_df = IntentPreprocessing.trim_df(df=sentence_df,
                                                  column_name=input_column,
                                                  new_column_name=output_column)

        # 用特定字元切開
        jieba = Get_JiebaSegmentor()
        sentence_df = IntentPreprocessing.cut_to_word_df(df=sentence_df,
                                                         column_name=input_column,
                                                         new_column_name=output_column,
                                                         js=jieba,
                                                         split_word=" ")

        return sentence_df

    def mapping_setting(self, sentence_df, input_column, output_column):
        """

        :param sentence_df:
        :param input_column:
        :param output_column:

        :return:
        """

        # 如果有指定就使用指定的mapping
        if self.mapping and self.mapping_name:
            pass

        else:
            # self.logger.debug('not mapping and mapping_name')

            # mapping & mapping_name 取出來
            mapping_dict = {}
            for index, row in sentence_df.iterrows():
                if mapping_dict.has_key(row['skill_id']):
                    continue
                mapping_dict.update({row['skill_id']: row['skill_name']})

            self.mapping = mapping_dict.keys()
            self.mapping_name = mapping_dict.values()

        self.logger.debug(self.mapping)
        self.logger.debug(self.mapping_name)

        # 總共類別數
        self.num_classes = len(self.mapping)
        self.logger.debug('num_classes = {}'.format(self.num_classes))

        # skill_id 轉爲 index , keras traing model 的label欄位必須是數字
        sentence_df = IntentPreprocessing.get_mapping_index_df(df=sentence_df,
                                                               column_name=input_column,
                                                               new_column_name=output_column,
                                                               mapping=self.mapping)
        return sentence_df

    def feature_engineering(self, sentence_df, input_column, output_column,
                            method='tfidf',
                            model_param={}):
        """
        建立文字特徵
        :param sentence_df:
        :param input_column:
        :param output_column:
        :param method:
        :param model_param:
        :return:
        """
        # print("feature_engineering")

        method = method.lower()
        cut_words = sentence_df[input_column]

        # print(cut_words)

        if method == 'tfidf':
            """
            TF-IDF（詞頻/逆文檔頻率）是最流行的IR（信息檢索）技術之一
            用於分析單詞在文檔中的重要性。研究表明，83％的基於文本的推薦系統使用TF-IDF
            TF-IDF衡量文檔中文字的重要性
            例如，「the」在任何文檔中都是常用的，因此TF-IDF並不認為「the」對文檔的特性很重要
            相反，IT相關主題使用「python」，TF-IDF認為「python」是識別主題和類別的重要特徵詞
            """

            vectorizer = TfidfVectorizer(norm=None, stop_words=None, token_pattern=r"(?u)\b\w+\b")

            # 預測時須將新的單字加入重新計算
            if self.transformer:
                # vectorizer.vocabulary = self.transformer.vocabulary_
                x_train_feature = self.transformer.transform(cut_words)
            else:
                self.transformer = vectorizer.fit(cut_words)
                x_train_feature = self.transformer.transform(cut_words)

            # x_train_feature = self.transformer.transform(cut_words)t
            output_value = list(x_train_feature.toarray())

        elif method == 'count':
            """
            Bag of Words是文檔數據的表示模型
            它簡單地計算單詞在文檔中出現的次數
            Bag-of-Words通常通過權衡特殊單詞和相關術語來用於聚類，分類和主題建模
            """

            vectorizer = CountVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")

            # 預測時須將新的單字加入重新計算
            if self.transformer:
                vectorizer.vocabulary = self.transformer.vocabulary_

            self.transformer = vectorizer.fit(cut_words)
            x_train_feature = self.transformer.transform(cut_words)
            output_value = list(x_train_feature.toarray())

        elif method == 'word2vec':
            """
            Word2vec擅長對相似的單詞進行分組，並根據上下文對單詞的含義進行高度準確的猜測
            它內部有兩種不同的算法：CBoW（Continuous Bag-of-Words）和skip gram模型
            Skip Gram用於預測目標詞的上下文
            CBoW是從上下文預測目標詞
            """
            output_value = []
        else:
            output_value = []

        # np array 轉 dataframe series
        sentence_df[output_column] = output_value

        # # 預測時不用保存特徵處理器
        # if is_training:
        #     # 將特徵處理器存到fs, ML預測時必須使用
        #     self.save_vectorizer(feature_transformer_path, feature_transformer_name)

        return sentence_df
        # return x_train_feature

    def data_split(self, sentence_df, feature_column, target_index_column,
                   train_test_split_size, one_skill, other_skill_id, model_param={}):
        """
        切割訓練測試資料
        :param sentence_df:
        :param feature_column:
        :param target_index_column:
        :param train_test_split_size:
        :param one_skill:
        :param other_skill_id:
        :param model_param:
        :return:
        """

        # 隨機切割出測試資料
        train_test_split_size = round(train_test_split_size, 2)
        train_df = sentence_df.sample(frac=train_test_split_size, random_state=100)
        test_df = sentence_df[~sentence_df.index.isin(train_df.index)]

        # 參數錯誤 >> 預設將所有資料丟入當測試資料
        if train_test_split_size >= 1 or train_test_split_size <= 0:
            train_df = sentence_df
            test_df = sentence_df

        if one_skill:
            # 語料庫的不要丟入測試
            test_df = test_df[test_df['skill_id'] != other_skill_id]

        self.logger.debug(
            "train_test_split_size : {}, len(train_df) : {}, len(test_df) : {}".format(str(train_test_split_size),
                                                                                       len(train_df),
                                                                                       len(test_df)))

        x_train = np.array(train_df[feature_column].tolist())
        y_train = train_df[target_index_column]

        return x_train, y_train, train_df, test_df

    def save_transformer(self, path, name):
        """
        持久化特徵轉換器
        :param path:
        :param name:
        :return:
        """

        if self.transformer:
            with open(path + name + ".pickle", 'wb') as handle:
                pickle.dump(self.transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.warning("The vectorizer is not exist. Please build vectorizer first.")

    def get_num_classes(self):
        return self.num_classes

    def train_model(self, x_train, y_train):
        """
        模型訓練
        """

        if self.model and self.model_param:

            # self.logger.debug(self.model)
            # self.logger.debug(type(x_train))
            # self.logger.debug(x_train)
            # self.logger.debug(type(y_train))
            # self.logger.debug(y_train)

            self.model.fit(x_train, y_train)

        else:
            self.logger.warning('please build or load model first !!')

        return self.model

    def train_history_plt(self, train='loss', validation='val_loss'):
        """

        :param train:
        :param validation:
        :return:
        """
        return None

    def test_model(self, x_test, y_test):
        """
        模型測試
        """
        pass

    def evaluate_model(self, x_test, y_test):
        """
        模型評估
        """
        pass

    def load_model(self, model_path, model_name):
        """
        模型載入
        """

        self.model = load(model_path + "/" + model_name)

    def save_model(self, model_path, model_name):
        """
        模型保存
        """

        if self.model:
            path = model_path + model_name + '.joblib'
            dump(self.model, path)
            # self.save_log()

        else:
            self.logger.debug('please build or load model first !!')

    def delete_model(self, model_path, model_name):
        """
        模型刪除
        """
        pass

    def predict_result(self, test_df, sentence_column, feature_column, target_column=None, target_index_column=None,
                       batch_size=64, verbose=1, model_param={}):
        """
        模型預測
        """
        self.logger.debug('============ predict_result ============')

        x_test = np.array(test_df[feature_column].tolist())
        # print(x_test)

        # self.logger.debug(x_test)=
        self.logger.debug(self.get_model())
        y_predict_probability = self.get_model().predict_proba(x_test)
        self.logger.debug('self.get_model() ml')

        # self.logger.debug(y_predict_probability)

        # 將預測機率找出最大值 and 把index找出來
        predict_class = np.argmax(y_predict_probability, axis=1)

        # self.logger.debug(predict_class)

        # 轉回原本分類名稱
        y_predict_id = IntentPreprocessing.to_cat_name(predict_class, self.mapping)
        y_predict_name = IntentPreprocessing.to_cat_name(predict_class, self.mapping_name)
        y_predict = predict_class

        # self.logger.debug(y_predict_id, y_predict_name, y_predict)

        # 將預測結果轉成小數點後4位
        predict_arr = []
        for row in y_predict_probability:
            row_arr = []
            for item in row:
                row_arr.append(IntentPreprocessing.float_display(item, 4))
            predict_arr.append(row_arr)

        # 沒有要驗證資料, 不需要答案欄位, defalut = -1
        answer = [-1 for _ in range(len(test_df[sentence_column]))]
        answer_id = [-1 for _ in range(len(test_df[sentence_column]))]
        if target_index_column:
            answer = test_df[target_index_column]
        if target_column:
            answer_id = test_df[target_column]

        predict_df = pd.DataFrame({'sentence': test_df[sentence_column],
                                   'answer': answer,
                                   'answer_id': answer_id,
                                   'y_predict': y_predict,
                                   'y_predict_id': y_predict_id,
                                   'y_predict_name': y_predict_name,
                                   'y_predict_probability': predict_arr})

        # self.logger.debug(predict_df)

        return predict_df


# class IntentRandomForestModelParam:
#
#     def __init__(self, n_estimators=20, max_depth=[3, None], criterion=["gini"], num_classes=0):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.criterion = criterion
#         self.num_classes = num_classes


class IntentRandomForestModel(IntentMLModel):

    def build_model(self, param):
        """
        RandomForest
        :param param:
        :return:
        """
        n_estimators = param['n_estimators']
        max_depth = param['max_depth']
        criterion = param['criterion']

        rf_model = RandomForestClassifier(n_estimators=n_estimators)
        parameters = {"max_depth": max_depth,
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": criterion}

        self.model = RandomizedSearchCV(rf_model,
                                        param_distributions=parameters,
                                        n_iter=20)
        self.model_param = param


# class IntentLogisticRegressionModelParam:
#
#     def __init__(self, penalty=['l2'], c=[10], num_classes=0):
#         self.penalty = penalty
#         self.c = c
#         self.num_classes = num_classes


class IntentLogisticRegressionModel(IntentMLModel):

    def build_model(self, param):
        """
        LogisticRegression
        :param param:
        :return:
        """
        parameters = {'penalty': param['penalty'],
                      'C': param['c']}

        lr_model = LogisticRegression()
        self.model = GridSearchCV(lr_model, parameters, cv=5)
        self.model_param = param
