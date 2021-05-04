# encoding=UTF-8
# !flask/bin/python

import sys

from functools import wraps
from werkzeug.exceptions import BadRequest
from flask import request


def requires_helper_user_robot(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('camp_user_id') is None:
                raise BadRequest(missing_error + "camp_user_id")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('robot_id') is None:
                raise BadRequest(missing_error + "robot_id")

        if request.method in ['POST', 'PUT', 'PATCH']:

            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

            # PATCH 欄位選填不卡
            if request.method in ['POST', 'PUT']:
                if not 'min_confidence' in request.json:
                    raise BadRequest(missing_error + "min_confidence")
                if not 'new_model_id' in request.json:
                    raise BadRequest(missing_error + "new_model_id")

        if request.method in ['DELETE']:

            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_online(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['POST']:

            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'model_id' in request.json:
                raise BadRequest(missing_error + "model_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")
            if not 'online' in request.json:
                raise BadRequest(missing_error + "online")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_algorithm(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        # if request.method in ['GET']:
        #     if request.args.get('robot_id') is None:
        #         raise BadRequest(missing_error + "robot_id")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "modify_user": unicode,
                          "min_confidence": float}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_online_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "modify_user": unicode,
                          "online": bool}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence_set(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('robot_id') is None:
                raise BadRequest(missing_error + "robot_id")

        if request.method in ['POST', 'PUT', 'PATCH']:
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'sentence_set_id' in request.json:
                raise BadRequest(missing_error + "sentence_set_id")

        if request.method in ['DELETE']:
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence_set_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "sentence_set_id": unicode,
                          "modify_user": unicode}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('robot_id') is None:
                raise BadRequest(missing_error + "robot_id")

        if request.method in ['POST', 'PUT', 'PATCH']:
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'sentence_set_id' in request.json:
                raise BadRequest(missing_error + "sentence_set_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")
            if not 'sentence_list' in request.json:
                raise BadRequest(missing_error + "sentence_list")

            for index, value in enumerate(request.json["sentence_list"]):

                s_index = str(index)
                if not value.has_key("skill_id"):
                    raise BadRequest(missing_error + "sentence_list.skill_id - index:" + s_index)

                if request.method in ['POST', 'PUT']:

                    if not value.has_key("sentence"):
                        raise BadRequest(missing_error + "sentence_list.sentence - index:" + s_index)
                elif request.method == 'PATCH':

                    if not value.has_key("sentence_id"):
                        raise BadRequest(missing_error + "sentence_list.sentence_id - index:" + s_index)
                    if not value.has_key("sentence"):
                        raise BadRequest(missing_error + "sentence_list.sentence - index:" + s_index)

        if request.method in ['DELETE']:
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "modify_user": unicode,
                          "sentence_list": list,
                          "sentence_set_id": unicode}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence_log(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('robot_id') is None:
                raise BadRequest(missing_error + "robot_id")
            if request.args.get('model_id') is None:
                raise BadRequest(missing_error + "model_id")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_train_sentence_log_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "sentence_log_id": unicode}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_train_log(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            if request.args.get('robot_id') is None:
                raise BadRequest(missing_error + "robot_id")

        if request.method in ['POST', 'PUT', 'PATCH']:

            # must
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'model_id' in request.json:
                raise BadRequest(missing_error + "model_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

            # POST&PUT必填 , PATCH 選填不卡
            if request.method in ['POST', 'PUT']:

                if not 'sentence_set_id' in request.json:
                    raise BadRequest(missing_error + "sentence_set_id")
                if not 'algorithm' in request.json:
                    raise BadRequest(missing_error + "algorithm")
                if not 'sentence_max_len' in request.json:
                    raise BadRequest(missing_error + "sentence_max_len")
                if not 'tokenizer_id' in request.json:
                    raise BadRequest(missing_error + "tokenizer_id")
                if not 'mapping' in request.json:
                    raise BadRequest(missing_error + "mapping")
                if not 'mapping_name' in request.json:
                    raise BadRequest(missing_error + "mapping_name")
                if not 'status' in request.json:
                    raise BadRequest(missing_error + "status")
                if not 'train_test_size' in request.json:
                    raise BadRequest(missing_error + "train_test_size")

            # # update only
            # if request.method in ['PUT', 'PATCH']:
            #     if not 'train_id' in request.json:
            #         raise BadRequest(missing_error + "train_id")

        if request.method == 'DELETE':

            # must
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'model_id_list' in request.json:
                raise BadRequest(missing_error + "model_id_list")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_train_log_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "modify_user": unicode,
                          "sentence_set_id": unicode,
                          "algorithm": unicode,
                          "sentence_max_len": int,
                          "tokenizer_id": unicode,
                          "mapping": list,
                          "mapping_name": list,
                          "train_test_size": float,
                          "model_id_list": list,
                          "status": unicode}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_test_log(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['GET']:
            robot_id = request.args.get('robot_id')
            if robot_id is None:
                raise BadRequest(missing_error + "robot_id")

        if request.method in ['POST', 'PUT', 'PATCH']:

            # must
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'model_id' in request.json:
                raise BadRequest(missing_error + "model_id")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

            # PATCH 欄位選填不卡
            if request.method in ['POST', 'PUT']:

                if not 'total_count' in request.json:
                    raise BadRequest(missing_error + "total_count")
                if not 'correct_count' in request.json:
                    raise BadRequest(missing_error + "correct_count")
                if not 'accuracy_score' in request.json:
                    raise BadRequest(missing_error + "accuracy_score")
                if not 'precision_score' in request.json:
                    raise BadRequest(missing_error + "precision_score")
                if not 'recall_score' in request.json:
                    raise BadRequest(missing_error + "recall_score")
                if not 'f1_score' in request.json:
                    raise BadRequest(missing_error + "f1_score")
                if not 'img' in request.json:
                    raise BadRequest(missing_error + "img")
                if not 'error_evaluate_sentence' in request.json:
                    raise BadRequest(missing_error + "error_evaluate_sentence")

        if request.method == 'DELETE':

            # must
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "robot_id")
            if not 'model_id_list' in request.json:
                raise BadRequest(missing_error + "model_id_list")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_test_log_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        # if request.args:
        #     check_dict = {"robot_id": unicode,
        #                   "model_id": unicode}
        #
        #     for key, value in check_dict.items():
        #         if request.args.get(key):
        #             if not isinstance(request.args.get(key), value): raise BadRequest(type_error + key)

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "modify_user": unicode,
                          "total_count": int,
                          "correct_count": int,
                          "accuracy_score": float,
                          "precision_score": float,
                          "recall_score": float,
                          "f1_score": float,
                          "img": unicode,
                          "error_evaluate_sentence": list,
                          "model_id_list": list}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)
    return decorated


def requires_helper_intent_model_training(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        missing_error = "缺少參數 : "

        if request.method in ['POST']:
            if not 'robot_id' in request.json:
                raise BadRequest(missing_error + "total_count")
            if not 'model_id' in request.json:
                raise BadRequest(missing_error + "model_id")
            if not 'sentence_set_id' in request.json:
                raise BadRequest(missing_error + "sentence_set_id")
            if not 'algorithm' in request.json:
                raise BadRequest(missing_error + "algorithm")
            if not 'modify_user' in request.json:
                raise BadRequest(missing_error + "modify_user")
            if not 'train_test_split_size' in request.json:
                raise BadRequest(missing_error + "train_test_split_size")
            # if not 'sentence_max_len' in request.json:
            #     raise BadRequest(missing_error + "sentence_max_len")

        return f(*args, **kwargs)

    return decorated


def requires_helper_intent_model_training_type(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        type_error = "參數型別錯誤 : "

        if request.json:
            check_dict = {"robot_id": unicode,
                          "model_id": unicode,
                          "modify_user": unicode,
                          "sentence_set_id": unicode,
                          "algorithm": unicode,
                          "sentence_max_len": int}
                          # "train_test_split_size": float}

            for key, value in check_dict.items():
                if key in request.json:
                    if not isinstance(request.json[key], value): raise BadRequest(
                        type_error + key)

        return f(*args, **kwargs)
    return decorated