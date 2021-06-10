# encoding=UTF-8
# !flask/bin/python


from abc import abstractmethod, ABCMeta

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
import keras.layers as L
import keras
from keras.layers.core import Activation, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, Reshape, Concatenate
from keras.layers import Conv2D, MaxPool2D, Embedding, GRU
from keras.models import Model

import pickle
import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MyEnv import Get_MyEnv
from Config_Helper import Get_HelperConfig
from Config_Format import Get_FormatConfig
import logging
from NLP_IntentModel import IntentModel
from NLP_IntentPreprocessing import IntentPreprocessing, IntentPreprocessing_Keras
from NLP_JiebaSegmentor import Get_JiebaSegmentor


class IntentDLModel:
    __metaclass__ = ABCMeta

    def __init__(self):

        # log
        # 系統log只顯示error級別以上的
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # 自訂log
        self.logger = logging.getLogger('IntentDLModel.py')
        self.logger.setLevel(logging.DEBUG)

        config = Get_HelperConfig()
        self.HELPER_KEYSPACE = Get_MyEnv().env_helper_keyspace
        self.HELPER_ERROR_LOG_TABLE = config.HELPER_ERROR_LOG_TABLE
        self.HELPER_INTENT_MODEL_TABLE = config.HELPER_INTENT_MODEL_TABLE
        self.HELPER_INTENT_TRAIN_SENTENCE_TABLE = config.HELPER_INTENT_TRAIN_SENTENCE_TABLE
        self.HELPER_INTENT_TRAIN_LOG_TABLE = config.HELPER_INTENT_TRAIN_LOG_TABLE
        self.HELPER_INTENT_TEST_LOG_TABLE = config.HELPER_INTENT_TEST_LOG_TABLE

        config = Get_FormatConfig()
        self.DATE_FORMAT_NORMAL = config.DATE_FORMAT_NORMAL

        self.model = None
        self.train_history = None
        self.model_param = None
        self.sentence_set_id = None
        self.mapping = None
        self.mapping_name = None
        self.num_classes = 0
        self.transformer = None

        self.algorithm_type = "DL"

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
        """
        訓練資訊
        :return:
        """
        self.logger.debug(self.model.summary())

    def get_model(self):

        if self.model is None:
            self.logger.warning('please train model first !!')

        return self.model

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
                                                         split_word=",")

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
                            method='tfidf', is_training=False,
                            model_param={}):
        """
        建立文字特徵
        :param sentence_df:
        :param input_column:
        :param output_column:
        :param method:
        :param is_training:
        :param model_param:
        :return:
        """

        self.logger.debug('feature_engineering')

        sentence_max_len = model_param['sentence_max_len']

        # fix parameters
        # 標點符號過濾
        WORD_FILTERS = '!"#$&()*+,-./:;<=>?@[\\]^_{|}~\t\n'
        # 字典數量 1+最大單字數
        NUM_WORDS = 500

        cut_words = sentence_df[input_column]
        # # cut_words = train_sentence_df['cut_words']

        # 有指定tokenizer的話就直接使用
        if self.transformer:
            # 新的單字不用加到字典
            pass

        else:
            # 使用Keras Tokenizer進行斷詞
            transformer = Tokenizer(filters=WORD_FILTERS,
                                    num_words=NUM_WORDS,
                                    split=",")

            # 斷詞後全部合併丟到fit_on_texts,組出字典
            transformer.fit_on_texts(cut_words)
            self.transformer = transformer

        # 轉成向量
        seq = IntentPreprocessing.texts_to_sequences(self.transformer,
                                                     cut_words)

        # 同義長度
        self.logger.debug('sentence_max_len : {}'.format(sentence_max_len))

        # np array 轉 dataframe series
        sentence_df[output_column] = IntentPreprocessing_Keras.pad_sequences(seq, sentence_max_len).tolist()

        # # 預測時不用保存特徵處理器
        # if is_training:
        #     # 將字典存到fs, DL預測時必須使用
        #     self.save_tokenizer(feature_transformer_path, feature_transformer_name)

        return sentence_df

        # return IntentPreprocessing.pad_sequences(seq, sentence_max_len)

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
        持久化字典
        :param path:
        :param name:
        :return:
        """

        if self.transformer:
            with open(path + name + ".pickle", 'wb') as handle:
                pickle.dump(self.transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.warning("The tokenizer is not exist. Please build tokenizer first.")

    def get_num_classes(self):

        return self.num_classes

    def get_train_history(self):

        if self.train_history is None:
            self.logger.warning('please train model first !!')

        return self.train_history

    def train_model(self, x_train, y_train):

        if self.model and self.model_param:

            batch_size = self.model_param['batch_size']
            epochs = self.model_param['epochs']
            train_ratio = self.model_param['train_ratio']
            early_stop = self.model_param['early_stop']
            optimizer = self.model_param['optimizer']
            loss = self.model_param['loss']

            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=['accuracy'])

            callbacks = []
            if early_stop > 0:
                earlystopping = EarlyStopping(monitor='val_loss', patience=20)
                callbacks.append(earlystopping)

            # if check_point:
            #     checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
            #     callbacks.append(checkpoint)

            self.train_history = self.model.fit(x=x_train, y=y_train,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                validation_split=1 - train_ratio,
                                                callbacks=callbacks)

            # 輸出loss與acc到日誌
            for i in range(len(self.train_history.history["acc"])):
                str_log = str(i + 1) + " Epoch " + "-loss: " + str(
                    self.train_history.history["loss"][i]) + " -acc: " + str(
                    self.train_history.history["acc"][i]) + " -val_loss: " + str(
                    self.train_history.history["val_loss"][i]) + " -val_acc: " + str(
                    self.train_history.history["val_acc"][i])
                self.logger.info(str_log)

            self.logger.debug('final val_loss : {}'.format(self.train_history.history['val_loss'][-1:]))
            self.logger.debug('final val_acc : {}'.format(self.train_history.history['val_acc'][-1:]))

        else:
            self.logger.warning('please build or load model first !!')

        return self.model

    def train_history_plt(self, train='loss', validation='val_loss'):
        """

        :param train:
        :param validation:
        :return:
        """

        if self.train_history:
            plt.plot(self.train_history.history[train])
            plt.plot(self.train_history.history[validation])
            plt.title('Train History')
            plt.ylabel(train)
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            return plt
            # plt.show()
        else:
            self.logger.warning('please train model first !!')
            return None

    # def accuracy(self):
    #
    #     if self.model_history:
    #         val_acc = self.model_history.history['val_acc'][-1:]
    #         val_loss = self.model_history.history['val_loss'][-1:]
    #
    #         return val_acc, val_loss
    #     else:
    #         return self.logger.warning('please build or load model first !!')

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

        self.model = load_model(model_path + "/" + model_name)

    def save_model(self, model_path, model_name):
        """
        模型保存
        """

        if self.model:
            self.model.save(model_path + model_name + '.h5')
            # self.save_log()

        else:
            self.logger.debug('please build or load model first !!')

    def save_image(self, path, name):
        """
        模型保存
        """

        if self.model:
            pass

        else:
            self.logger.debug('please build or load model first !!')

    def delete_model(self, model_path, model_name):
        """
        模型刪除
        """
        pass

    def predict_result(self, test_df, sentence_column, feature_column,
                       target_column=None, target_index_column=None, model_param={}):
        """
        模型預測
        """
        self.logger.debug('============ predict_result ============')

        x_test = np.array(test_df[feature_column].tolist())

        # self.logger.debug(x_test)

        self.logger.debug(self.get_model())
        # self.get_model()._make_predict_function()
        verbose = model_param.get('model_param', 1)
        y_predict_probability = self.get_model().predict(x_test,
                                                         batch_size=model_param['batch_size'],
                                                         verbose=verbose)
        self.logger.debug('self.get_model() dl')

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

        # 預測結果
        predict_df = pd.DataFrame({'sentence': test_df[sentence_column],
                                   'answer': answer,
                                   'answer_id': answer_id,
                                   'y_predict': y_predict,
                                   'y_predict_id': y_predict_id,
                                   'y_predict_name': y_predict_name,
                                   'y_predict_probability': predict_arr})

        # self.logger.debug(predict_df)

        return predict_df


class IntentOnedCnnModel(IntentDLModel):

    def build_model(self, param):
        """
        OnedCnn
        使用長度不同的filter 對文本矩陣進行卷積，filter的寬度等於詞向量的長度，
        然後使用max-pooling 對每一filter提取的向量進行操作，
        最後每一個filter對應一個數字，把這些filter拼接起來，就得到了一個表徵該句子的向量
        :param param:
        :return:
        """

        num_classes = param['num_classes']
        vocab_size = param['vocab_size']
        sentence_max_len = param['sentence_max_len']
        embedding_output_dim = param['embedding_output_dim']
        kernel_size = param['kernel_size']
        filters = param['filters']

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_output_dim,
                            input_length=sentence_max_len))

        model.add(Dropout(0.75))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        #     model.add(Dense(64))
        #     model.add(Dropout(0.5))
        model.add(Dense(32))
        model.add(Dropout(0.5))

        model.add(Activation('relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # model.summary()
        self.model = model
        self.model_param = param


class IntentTwodCnnModel(IntentDLModel):

    def build_model(self, param):
        """
        TwodCnn
        :return:
        """

        num_classes = param['num_classes']
        vocab_size = param['vocab_size']
        sentence_max_len = param['sentence_max_len']
        embedding_output_dim = param['embedding_output_dim']
        filter_sizes = param['filter_sizes']
        num_filters = param['num_filters']
        drop_out = param['drop_out']

        inputs = Input(shape=(sentence_max_len,), dtype='int32')
        embedding = Embedding(input_dim=vocab_size + 1,
                              output_dim=embedding_output_dim,
                              input_length=sentence_max_len,
                              trainable=False)(inputs)

        reshape = Reshape((sentence_max_len, embedding_output_dim, 1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_output_dim),
                        padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_output_dim),
                        padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_output_dim),
                        padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(sentence_max_len - filter_sizes[0] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sentence_max_len - filter_sizes[1] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sentence_max_len - filter_sizes[2] + 1, 1),
                              strides=(1, 1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop_out)(flatten)
        preds = Dense(num_classes, activation='softmax')(dropout)

        # this creates a model that includes inputs and outputs
        model = Model(inputs=inputs, outputs=preds)

        # model.summary()
        self.model = model
        self.model_param = param


class IntentTextCnnModel(IntentDLModel):

    def build_model(self, param):
        """
        TextCnn
        :param param:
        :return:
        """

        num_classes = param['num_classes']
        vocab_size = param['vocab_size']
        sentence_max_len = param['sentence_max_len']
        embedding_output_dim = param['embedding_output_dim']
        drop_out = param['drop_out']
        l2_reg_lambda = param['l2_reg_lambda']

        self.logger.debug('Build model...')
        input_x = L.Input(shape=(sentence_max_len,), name='input_x')

        filter_sizes = [2, 3]
        num_filters = 256
        # embedding layer
        # if embedding_matrix is None:
        embedding = L.Embedding(vocab_size, embedding_output_dim, name='embedding')(input_x)
        # else:
        #    embedding = L.Embedding(vocab_size, embedding_size, weights=[embedding_matrix], name='embedding')(input_x)
        expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
        # embedding_chars = K.expand_dims(embedding, -1)    # 4D tensor [batch_size, seq_len, embeding_size, 1] seems like a gray picture
        embedding_chars = L.Reshape(expend_shape)(embedding)

        # conv->max pool
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            conv = L.Conv2D(filters=num_filters,
                            kernel_size=[filter_size, embedding_output_dim],
                            strides=1,
                            padding='valid',
                            activation='relu',
                            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                            bias_initializer=keras.initializers.constant(value=0.1),
                            name=('conv_%d' % filter_size))(embedding_chars)
            # self.logger.debug("conv-%d: " % i, conv)
            max_pool = L.MaxPool2D(pool_size=[sentence_max_len - filter_size + 1, 1],
                                   strides=(1, 1),
                                   padding='valid',
                                   name=('max_pool_%d' % filter_size))(conv)
            pooled_outputs.append(max_pool)
            # self.logger.debug("max_pool-%d: " % i, max_pool)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = L.Concatenate(axis=3)(pooled_outputs)
        h_pool_flat = L.Reshape([num_filters_total])(h_pool)
        # add dropout
        dropout = L.Dropout(drop_out)(h_pool_flat)

        # output layer
        output = L.Dense(num_classes,
                         kernel_initializer='glorot_normal',
                         bias_initializer=keras.initializers.constant(0.1),
                         activation='softmax',
                         name='output')(dropout)
        model = keras.models.Model(inputs=input_x, outputs=output)
        # model.summary()
        self.model = model
        self.model_param = param


class IntentGRUModel(IntentDLModel):

    def build_model(self, param):
        num_classes = param['num_classes']
        vocab_size = param['vocab_size']
        sentence_max_len = param['sentence_max_len']
        embedding_output_dim = param['embedding_output_dim']
        drop_out = param['drop_out']

        model = Sequential()

        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_output_dim,
                            input_length=sentence_max_len))
        model.add(Dropout(drop_out))
        #     model.add(Flatten())
        #     model.add(SimpleRNN(embedding_output_dim, return_sequences=True))-
        #     model.add(Dropout(0.5))
        #     model.add(SimpleRNN(embedding_output_dim, return_sequences=True))
        #     model.add(Dropout(0.5))
        #     model.add(SimpleRNN(embedding_output_dim, return_sequences=True))
        #     model.add(Dropout(0.5))
        #     model.add(SimpleRNN(embedding_output_dim))
        #     model.add(Dropout(0.5))

        #     model.add(Dropout(0.2))
        model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))

        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()
        self.model = model
        self.model_param = param
