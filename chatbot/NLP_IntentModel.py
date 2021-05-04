# encoding=UTF-8
# !flask/bin/python


from abc import abstractmethod, ABCMeta

from envCassandraDAO import envCassandraDAO
from MyEnv import Get_MyEnv
from Config_Helper import Get_HelperConfig
from Config_Format import Get_FormatConfig
import logging


class IntentModel:
    __metaclass__ = ABCMeta

    def __init__(self):
        # log
        # 系統log只顯示error級別以上的
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # 自訂log
        self.logger = logging.getLogger('NLP_IntentModelFactory.py')
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

        self.c_dao = envCassandraDAO(Get_MyEnv().env_cassandra)

        self.model = None
        self.train_history = None
        self.model_param = None
        self.sentence_set_id = None
        self.mapping = None
        self.mapping_name = None
        self.num_classes = 0
        self.transformer = None

    def __repr__(self):
        return "{class_name}".format(class_name=self.__class__.__name__)

    # @abstractmethod
    # def create_model(self):
    #     pass
    #
    # @abstractmethod
    # def get_model(self):
    #     pass
    #
    # @abstractmethod
    # def preprocessing(self):
    #     pass
    #
    # @abstractmethod
    # def feature_engineering(self):
    #     pass
    #
    # @abstractmethod
    # def train_model(self):
    #     pass
    #
    # @abstractmethod
    # def load_model(self):
    #     pass
    #
    # @abstractmethod
    # def save_model(self):
    #     pass
    #
    # @abstractmethod
    # def delete_model(self):
    #     pass