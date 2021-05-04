# encoding=UTF-8
# !flask/bin/python

import logging
from datetime import datetime
import pickle
import numpy as np
import json
import requests
import pandas as pd
import decimal

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


class IntentPreprocessing(object):

    @staticmethod
    def create_tokenizer(_df, _word_filters, _num_words):
        """
        將句子斷詞, 組出字典
        :param _df:
        :param _word_filters:
        :param _num_words:
        :return:
        """

        tokenizer = Tokenizer(filters=_word_filters,
                              num_words=_num_words,
                              split=",")

        # 斷詞後全部合併丟到fit_on_texts,組出字典
        tokenizer.fit_on_texts(_df)

        return tokenizer

    @staticmethod
    def save_tokenizer(_tokenizer, path, name):
        """
        save 字典
        :param _tokenizer:
        :param path:
        :param name:
        :return:
        """

        with open(path + name + '.pickle', 'wb') as handle:
            pickle.dump(_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_tokenizer(path, name):
        """
        load 字典
        :param path:
        :param name:
        :return:
        """

        with open(path + name + '.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        return tokenizer

    @staticmethod
    def texts_to_sequences(tokenizer, cut_words):
        """
        將文字轉為數字序列
        :param tokenizer:
        :param cut_words:
        :return:
        """

        seq = tokenizer.texts_to_sequences(cut_words)
        #     print train_seq_intent

        return seq

    @staticmethod
    def pad_sequences(seq, max_len=15):
        """
        截長補短，讓文字所產生的數字序列長度一樣
        :param seq:
        :param max_len:
        :return:
        """

        seq = sequence.pad_sequences(seq, maxlen=max_len)
        #     print data.shape
        return seq

    @staticmethod
    def to_one_hot(_y_all):
        """
        label one hot encoding
        :param y_all:
        :return:
        """

        return np_utils.to_categorical(_y_all)

    @staticmethod
    def to_cat_name(_x, _mapping_list):
        """
        training的類別對應表
        預測數字轉文字名稱
        :param _x:
        :param _mapping_list:
        :return:
        """
        mapping_name_list = []
        for item in _x:
            mapping_name_list.append(_mapping_list[int(item)])
        return mapping_name_list

    @staticmethod
    def category_mapping(_df):
        """
        index to categories mapping
        :param _df:
        :return:

        Index([u'air', u'coffee', u'leave', u'po', u'point', u'qcall', u'traffic', u'weather'],
        dtype='object')
        """

        mapping = _df.target.astype('category').cat.categories
        return mapping

    @staticmethod
    def shuffle(_df):
        """
        打散資料
        :param _df:
        :return:
        """

        return _df.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def get_mapping_loc(mapping, target):
        """
        取得mapping index
        :param mapping:
        :param target:
        :return:
        """

        return mapping.get_loc(target)

    @staticmethod
    def cut_to_word(_js, _s):
        """
        將句子使用逗號分割, 方便 tokenizer斷詞
        :param _js:
        :param _s:
        :return:
        """
        w_df = _js.lcut(_s, return_type='pandas')
        combie = ''
        for i, w in w_df.iterrows():
            combie = combie + w
            if i < len(w_df) - 1:
                combie = combie + ','

        return combie

    @staticmethod
    def float_display(_f, _float_display):
        """
        顯示小數點n位
        :param _f:
        :param _float_display:
        :return:
        """

        return round(_f, _float_display)