# encoding=UTF-8
# !flask/bin/python


from abc import abstractmethod, ABCMeta
import os
import torch

from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange

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
from NLP_IntentPreprocessing import IntentPreprocessing


class IntentBERTModel:
    __metaclass__ = ABCMeta

    def __init__(self):

        # log
        # 系統log只顯示error級別以上的
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # 自訂log
        self.logger = logging.getLogger('IntentBERTModel.py')
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

        self.algorithm_type = "BERT"

        # specify GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            torch.cuda.get_device_name(0)

        if torch.cuda.is_available():
            print("=========== torch.cuda.is_available() == True !!! ===========")
        else:
            print("=========== torch.cuda.is_available() == False !!! ===========")

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
        self.logger.debug(self.model)

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

        sentence_df[output_column] = sentence_df[input_column]

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
                            is_training=False, model_param={}):
        """
        建立文字特徵
        :param sentence_df:
        :param input_column:
        :param output_column:
        :param is_training:
        :param model_param:
        :return:
        """

        self.logger.debug('feature_engineering')

        sentence_max_len = model_param['sentence_max_len']

        sentence_df[input_column] = sentence_df[input_column].apply(lambda s: "[CLS] " + s + " [SEP]")

        bert_base = model_param['bert_base']
        bert_tokenizer_path = model_param['bert_tokenizer_path']
        attention_masks_column = model_param['attention_masks_column']

        # 有指定tokenizer的話就直接使用
        if self.transformer:
            pass

        else:

            # 取得此預訓練模型所使用的 tokenizer
            # 指定繁簡中文 BERT-BASE 預訓練模型

            ## download tokenizer
            # tokenizer = BertTokenizer.from_pretrained(bert_base)

            ## local tokenizer
            tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)

            self.transformer = tokenizer

        sentence = list(sentence_df[input_column])

        # Restricting the max size of Tokens to 512(BERT doest accept any more than this)
        tokenized_texts = list(map(lambda t: self.transformer.tokenize(t)[:510], sentence))
        self.logger.debug("Tokenize the first sentence:")
        self.logger.debug(tokenized_texts[0])

        # Pad our input tokens so that everything has a uniform length
        input_pad = pad_sequences(list(map(self.transformer.convert_tokens_to_ids, tokenized_texts)),
                                  maxlen=sentence_max_len,
                                  dtype="int",
                                  truncating="post",
                                  padding="post").tolist()

        # self.logger.debug("pad_sequences:")
        # self.logger.debug(input_pad)

        # # test
        # # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        # input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="int", truncating="post", padding="post")

        # BERT is a MLM(Masked Language Model).We have to define its mask.
        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_pad:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # self.logger.debug("attention_masks:")
        # self.logger.debug(attention_masks)

        # 同義長度
        self.logger.debug('sentence_max_len : {}'.format(sentence_max_len))

        # np array 轉 dataframe series
        sentence_df[output_column] = input_pad

        attention_masks_column = attention_masks_column
        sentence_df[attention_masks_column] = attention_masks

        return sentence_df

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

        attention_masks_column = model_param['attention_masks_column']
        batch_size = model_param['batch_size']

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

        train_inputs = list(train_df[feature_column])
        validation_inputs = list(test_df[feature_column])
        train_labels = list(train_df[target_index_column])
        validation_labels = list(test_df[target_index_column])
        train_masks = list(train_df[attention_masks_column])
        validation_masks = list(test_df[attention_masks_column])

        # Convert all of our data into torch tensors, the required datatype for our model
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # Create an iterator of our data with torch DataLoader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=batch_size)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,
                                           sampler=validation_sampler,
                                           batch_size=batch_size)

        # self.logger.debug(train_dataloader)
        # self.logger.debug(validation_dataloader)

        return train_dataloader, validation_dataloader, train_df, test_df

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
            self.logger.warning("using pretrained tokenizer.")

    def get_num_classes(self):

        return self.num_classes

    def get_train_history(self):

        if self.train_history is None:
            self.logger.warning('please train model first !!')

        return self.train_history

    def train_model(self, train_dataloader, validation_dataloader):

        if self.model and self.model_param:

            batch_size = self.model_param['batch_size']
            epochs = self.model_param['epochs']
            no_decay = self.model_param['no_decay']

            # BERT fine-tuning parameters
            param_optimizer = list(self.model.named_parameters())

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=2e-5,
                                 warmup=.1)

            # Function to calculate the accuracy of our predictions vs labels
            def flat_accuracy(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return np.sum(pred_flat == labels_flat) / len(labels_flat)

            torch.cuda.empty_cache()

            # Store our loss and accuracy for plotting
            train_loss_set = []

            # self.logger.debug('~~~~~self.model')
            # self.logger.debug(self.model)

            # BERT training loop
            for _ in trange(epochs, desc="Epoch"):

                ## TRAINING

                # Set our model to training mode
                self.model.train()
                # Tracking variables
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                # Train the data for one epoch
                for step, batch in enumerate(train_dataloader):
                    # Add batch to GPU
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch
                    # self.logger.debug('@@@@@@@@@')
                    # self.logger.debug(b_input_ids[0])
                    # self.logger.debug(b_input_mask[0])
                    # self.logger.debug(b_labels[0])
                    #
                    # self.logger.debug(len(b_input_ids))
                    # self.logger.debug(len(b_input_mask))
                    # self.logger.debug(len(b_labels))

                    # Clear out the gradients (by default they accumulate)
                    optimizer.zero_grad()
                    # Forward pass
                    loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

                    train_loss_set.append(loss.item())
                    # Backward pass
                    loss.backward()
                    # Update parameters and take a step using the computed gradient
                    optimizer.step()
                    # Update tracking variables
                    tr_loss += loss.item()
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1
                print("Train loss: {}".format(tr_loss / nb_tr_steps))

                ## VALIDATION

                # Put model in evaluation mode
                self.model.eval()
                # Tracking variables
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                # Evaluate data for one epoch
                for batch in validation_dataloader:
                    # Add batch to GPU
                    batch = tuple(t.to(self.device) for t in batch)
                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch
                    # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions
                        logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                        # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    eval_accuracy += tmp_eval_accuracy
                    nb_eval_steps += 1
                print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

            self.train_history = train_loss_set


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

            plt.figure(figsize=(15, 8))
            plt.title("Training loss")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.plot(self.train_history)
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

            model_to_save = self.model.module if hasattr(self.model,
                                                         'module') else self.model  # Only save the model it-self
            output_model_file = os.path.join(model_path, model_name + ".bin")
            # if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)


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
                       target_column=None, target_index_column=None,
                       model_param={}):
        """
        模型預測
        """
        self.logger.debug('============ predict_result ============')

        # self.logger.debug(len(test_df))

        validation_inputs = list(test_df[feature_column])
        validation_masks = list(test_df[model_param['attention_masks_column']])

        # Convert all of our data into torch tensors, the required datatype for our model
        validation_inputs = torch.tensor(validation_inputs)
        validation_masks = torch.tensor(validation_masks)

        # self.logger.debug(len(validation_inputs))
        # self.logger.debug(len(validation_masks))
        # self.logger.debug(len(validation_labels))

        """表示是新的句子預測(沒有target label) 而非 驗證資料"""
        if target_index_column:
            validation_labels = list(test_df[target_index_column])
            validation_labels = torch.tensor(validation_labels)
            validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        else:
            validation_data = TensorDataset(validation_inputs, validation_masks)

        validation_sampler = SequentialSampler(validation_data)

        # 每次load出batch_size
        validation_dataloader = DataLoader(validation_data,
                                           sampler=validation_sampler,
                                           batch_size=len(test_df))

        # self.logger.debug(self.get_model())

        logits = []
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            if target_index_column:
                b_input_ids, b_input_mask, b_labels = batch
            else:
                b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # """model predict"""
        # logits = self.model(validation_inputs, token_type_ids=None, attention_mask=validation_masks)

        y_predict_probability = []

        """ 
        Move logits and labels to CPU
        numpy does not support GPU
        """
        # logits = logits.detach().cpu().numpy()

        for ii, x in enumerate(logits):
            if torch.cuda.is_available():
                y_predict_probability.append(np.array(F.softmax(x).cpu().numpy()))
            else:
                y_predict_probability.append(np.array(F.softmax(x)))

            # self.logger.debug('self.get_model() transfer')

        # self.logger.debug(y_predict_probability)

        # 將預測機率找出最大值 and 把index找出來
        predict_class = np.argmax(y_predict_probability, axis=1)

        # self.logger.debug(predict_class)

        # 轉回原本分類名稱
        y_predict_id = IntentPreprocessing.to_cat_name(predict_class, self.mapping)
        y_predict_name = IntentPreprocessing.to_cat_name(predict_class, self.mapping_name)
        y_predict = predict_class
        # self.logger.debug('y_predict_id')
        # self.logger.debug(y_predict_id)
        # self.logger.debug(y_predict_name)
        # self.logger.debug(y_predict)
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


class IntentBertModel(IntentBERTModel):

    def build_model(self, param):
        num_classes = param['num_classes']
        bert_base = param['bert_base']
        bert_model = param['bert_model']

        model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_classes)
        # model = BertForSequenceClassification.from_pretrained(bert_base, num_labels=num_classes)
        if torch.cuda.is_available():
            model.cuda()

        # model.summary()
        self.model = model
        self.model_param = param
