# encoding=UTF-8
# !flask/bin/python


from abc import abstractmethod, ABCMeta

from NLP_IntentModel import IntentModel
# from NLP_IntentBertModel import *
from NLP_IntentDLModel import *
from NLP_IntentMLModel import *


class IntentModelFactory:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def __repr__(self):
        return "{class_name}".format(class_name=self.__class__.__name__)

    def create_model(self):
        pass


class IntentOnedCnnModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='1D CNN'))
        return IntentOnedCnnModel()


class IntentTwodCnnModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='2D CNN'))
        return IntentTwodCnnModel()


class IntentTextCnnModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='Text CNN'))
        return IntentTextCnnModel()


class IntentGRUModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='GRU'))
        return IntentGRUModel()


class IntentRandomForestModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='RandomForest'))
        return IntentRandomForestModel()


class IntentLogisticRegressionModelFactory(IntentModelFactory):

    def create_model(self):
        print('create model : {algorithm}...'.format(algorithm='LogisticRegression'))
        return IntentLogisticRegressionModel()

#
# class IntenBertModelFactory(IntentModelFactory):
#
#     def create_model(self):
#         print('create model : {algorithm}...'.format(algorithm='BERT'))
#         return IntentBertModel()

# factory = IntentTextCnnModelFactory()
# model = factory.create_model()
# model.build_model()
