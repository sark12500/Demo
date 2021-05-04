# -*- coding: utf-8 -*-
import sys
import os

# reload(sys)
# sys.setdefaultencoding('utf-8')
import abc
import pandas as pd
from MyEnv import Get_MyEnv
from MssqlDAO import MssqlDAO
import logging

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
# 自訂log
logger = logging.getLogger('NLP_ChatBot.py')
logger.setLevel(logging.DEBUG)


class Theme(metaclass=abc.ABCMeta):

    def __init__(self):
        self.theme = self.__class__.__name__.lower()
        self.query_table = "dcard_query_table_" + self.theme + "_chatbot"
        self.word_article_table = "dcard_word_article_" + self.theme + "_chatbot"

    def __repr__(self):
        return "{class_name}".format(class_name=self.__class__.__name__)

    def __str__(self):
        return "{class_name}".format(class_name=self.__class__.__name__.lower())


#     @abc.abstractmethod
#     def a(self):
#         print('hi')

class Food(Theme):

    def __init__(self):
        super(Food, self).__init__()


class Money(Theme):

    def __init__(self):
        super(Money, self).__init__()


class Beauty(Theme):

    def __init__(self):
        super(Beauty, self).__init__()


class Sport(Theme):

    def __init__(self):
        super(Sport, self).__init__()


class ThemeFactory:
    """
    工廠模式介面類
    :return:
    """

    def __init__(self):
        self.theme_dict = {
            "food": Food(),
            "money": Money(),
            "beauty": Beauty(),
            "sport": Sport()
        }

    def create(self, theme):
        logger.debug("create")
        logger.debug(theme)
        logger.debug(self.theme_dict.get(theme, None))
        return self.theme_dict.get(theme, None)


class ChatBot(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    def __repr__(self):
        return "{class_name}".format(class_name=self.__class__.__name__)

    def __str__(self):
        return "{class_name}".format(class_name=self.__class__.__name__.lower())


#     @abc.abstractmethod
#     def a(self):
#         print('hi')

#Person().behavior_in_morning(SoyMilk("IMei Sugarless"))

class FreeTalkChatBot(ChatBot):

    def __init__(self, tokenizer, tfidf_model=None):
        super(FreeTalkChatBot, self).__init__()
        self.tokenizer = tokenizer
        self.tfidf_model = tfidf_model
        self.dao = MssqlDAO(ip=Get_MyEnv().env_mssql_ip,
                            account=Get_MyEnv().env_mssql_account,
                            password=Get_MyEnv().env_mssql_password)

    #         self.free_talk_skill = FreeTalkSkill()

    def find_keyword(self, text):

        # 依照TFIDF給權重取關鍵字
        text = text.lower()

        words_df = self.tokenizer.pseg_lcut_combie_num_eng(text)
        cut_len = len(words_df)

        v_tag = ['v', 'vd', 'vg', 'vi', 'vn', 'vq', 'vt']
        n_tag = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']
        a_tag = ['a', 'ad', 'ag', 'an']
        m_tag = ['m', 'eng', 'm+eng']

        tags = v_tag + n_tag + a_tag + m_tag

        if self.tfidf_model:
            tfidf_words_df = self.tfidf_model.predict(text,
                                                      min_tfidf=0)

            # if word is not in tfidf model, word will be removed. So I need to compare with jieba cut legth.
            if len(tfidf_words_df) == cut_len:

                # 增加index欄位當權重用
                words_df = tfidf_words_df.reset_index()
                word_len = len(words_df)

                def set_weight_udf(row):
                    return word_len - row['index']

                words_df['weight'] = words_df.apply(set_weight_udf, axis=1)

            else:

                # 依照詞性取關鍵字
                # words_df = jieba.pseg_lcut_combie_num_eng(text)

                words_df = words_df[words_df['sp'].isin(tags)]
                words_df = words_df.rename(columns={"word": "keyword"})
                words_df['tfidf'] = 1
                words_df['weight'] = 1

        else:

            # 依照詞性取關鍵字
            # words_df = jieba.pseg_lcut_combie_num_eng(text)

            words_df = words_df[words_df['sp'].isin(tags)]
            words_df = words_df.rename(columns={"word": "keyword"})
            words_df['tfidf'] = 1
            words_df['weight'] = 1

        logger.debug(words_df)

        return words_df

    def query_word_article(self, keyword_df, word_article_table):
        """
        取得關鍵字對應的文章
        """

        word_list_str = "'" + "','".join(list(keyword_df['keyword'])) + "'"

        article_df = pd.DataFrame({})
        # by參數查詢版, * 表示所有版都查詢
        # # if category == ["*"]:
        # if chatbot_id == "*":
        #     pass
        #
        # else:

        # for cc in category:
        if word_article_table:
            sql = ("select article_id, article_title, article_topics, article_like_count, " +
                   " article_comment_count, article_created_at,  article_updated_at , " +
                   "count(article_id) as count from " +
                   word_article_table +
                   " where keyword in (" + word_list_str + ") group by article_id , " +
                   " article_title, article_topics, article_like_count," +
                   " article_like_count,  article_comment_count, article_created_at, " +
                   " article_updated_at order by count desc ")
            #             logger.debug(sql)
            single_df = self.dao.execCQLSelectToPandasDF(Get_MyEnv().env_mssql_db, sql)
            article_df = article_df.append(single_df)

        logger.debug(article_df)

        return article_df

    def article_similarly_calculate(self, df, by_column):
        """
        取得最大關鍵字出現數量的文章, 表示與input_text最像
        """
        df = df.sort_values(by=[by_column], ascending=False).reset_index(drop=True)

        max_count = df[by_column][0]
        logger.debug("max_{} : {}".format(by_column, max_count))

        df = df[df[by_column] == max_count]
        logger.debug('len : {}'.format(len(df)))

        return df.reset_index(drop=True)

    def article_filter_by_tfidf(self, df, keyword_df):
        """
        關鍵字match數量相同時, 依照tfidf weight排序
        """

        # 輸入文字的單字與weight組成的字典
        word_weight_dict = dict(zip(keyword_df.keyword, keyword_df.weight))
        keyword_list = list(keyword_df['keyword'])

        def calculate_weight_udf(row):

            weight = 0
            for ww in keyword_list:
                if ww in row['article_title'] or ww in row['article_topics']:
                    weight = weight + word_weight_dict[ww]

            return weight

        df['weight'] = df.apply(calculate_weight_udf, axis=1)

        return df.sort_values(by=['weight'], ascending=False)

    def query_comment(self, df, query_table, top_n=None):
        """
        取得回文
        """

        if top_n and top_n < 1:
            top_n = 1

        #     logger.debug(top_n)

        article_id_list = [str(x) for x in list(df['article_id'])]
        article_id_list_str = "'" + "','".join(article_id_list) + "'"

        comment_df = pd.DataFrame({})
        # for cc in category:

        if top_n:
            sql = ("select top(" + str(top_n) + ")* from " + query_table +
                   " where article_id in (" + article_id_list_str + ") order by comment_like_count desc")
        else:

            sql = ("select * from " + query_table +
                   " where article_id in (" + article_id_list_str + ") order by comment_like_count desc")
        #         logger.debug(sql)

        single_df = self.dao.execCQLSelectToPandasDF(Get_MyEnv().env_mssql_db, sql)
        comment_df = comment_df.append(single_df)

        return comment_df

    def talk(self, theme, input_text, top_n_article=10, top_n_comment=20, sort_by='random'):

        logger.debug("Theme : " + str(theme))
        logger.debug("input_text : " + input_text)
        logger.debug("input_text : " + sort_by)

        # if len(category) == 0:
        #     return None, '......沒有對應的版'

        # 找出input_text關鍵字
        input_text = input_text.lower()

        keyword_df = self.find_keyword(input_text)

        logger.debug(keyword_df)

        # 取出input_text關鍵字與相關 的文章title
        article_df = self.query_word_article(keyword_df, word_article_table=theme.word_article_table)

        if len(article_df) == 0:
            return None, '......沒有關鍵字對應的文章'

        # 取得最大關鍵字出現數量的文章, 表示與input_text最像
        article_df = self.article_similarly_calculate(article_df, 'count')

        # 關鍵字match數量相同時, 依照tfidf weight排序
        article_df = self.article_filter_by_tfidf(article_df, keyword_df)

        # logger.debug(article_df.head())

        # 取得最大weight, 表示與input_text最像
        article_df = self.article_similarly_calculate(article_df, 'weight')

        logger.debug("len(print(article_df)) : {}".format(len(article_df)))

        # 文章過多時, 近期受歡迎文章優先取出top n
        if len(article_df) > top_n_article:
            article_df = article_df.sort_values(by=['article_comment_count', 'article_created_at'],
                                                ascending=False).reset_index(drop=True)[:top_n_article]

        logger.debug(article_df)

        # 取得回文
        comment_df = self.query_comment(article_df, theme.query_table)

        if len(comment_df) == 0:
            return None, '......沒有回文'

        #     comment_df = comment_df[comment_df.comment_content.str.contains('原po') == False]
        #     comment_df = comment_df[comment_df.article_topics.str.contains('食記') == False]

        if sort_by == "random":
            comment_df = comment_df.sample(frac=1).reset_index(drop=True)
        else:
            comment_df = comment_df.sort_values(by=[sort_by],
                                                ascending=False).reset_index(drop=True)

        comment_df = comment_df[:top_n_comment]
        comment_df = comment_df[
            ['article_id', 'article_title', 'article_topics',
             'comment_content', 'comment_like_count', 'comment_created_at']]

        return comment_df, 'success'
