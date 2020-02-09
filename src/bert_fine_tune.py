pip install tensorflow_hub
pip install bert-tensorflow
import keras
import pandas as pd
from time import time
import datetime
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import bert
from bert import tokenization

MAX_SEQ_LENGTH = 100
BERT_VOCAB = '/s3mnt/Bert/uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = '/s3mnt/Bert/uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = '/s3mnt/Bert/uncased_L-12_H-768_A-12/bert_config.json'
DOC_PATH = '/s3mnt/quora_duplicate_questions.tsv'
class():
    def __init__(self,BERT_VOCAB,BERT_INIT_CHKPNT,BERT_CONFIG,DOC_PATH,MAX_SEQ_LENGTH):
        self.BERT_VOCAB = BERT_COVAB
        self.BERT_INIT_CHKPNT = BERT_INIT_CHKPNT
        self.BERT_CONFIG =BERT_CONFIG
        self.DOC_PATH = DOC_PATH
        self.max_seq_length = MAX_SEQ_LENGTH
        
    def load_dataset(self):
        df = pd.read_csv(self.DOC_PATH,header = 0, sep = ',',skiprows = skip_rows).dropna()
        left, right, label = df['question1'].tolist(), df['question2'].tolist(), df['is_duplicate'].tolist()
        return left, right, label
            
        
    def createTokenizer(self)
        tokenizer = tokenization.FullTokenizer(vocab_file=self.BERT_VOCAB, do_lower_case=True)
        return tokenizer

    def get_masks(self，tokens):
    """Mask for padding"""
        if len(tokens)>self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (self.max_seq_length - len(tokens))


    def get_segments(self，tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
            current_segment_id = 1
        return segments + [0] * (self.max_seq_length - len(tokens))


    def get_ids(self，tokens, tokenizer):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_seq_length-len(token_ids))
        return input_ids

    def add_tokens(self，questionList):
        return [["[CLS]"] + tokenizer.tokenize(question) + ["[SEP]"] for question in left]
    
    def get_input_matirx(self，left_tokens):
        input_ids_left, input_masks_left, segment_ids_left = [], [], []
        for i in range(len(left_tokens)):
            input_id = self.get_ids(left_tokens[i],tokenizer)
            input_mask = self.get_masks(left_tokens[i])
            segment_id = self.get_segments(left_tokens[i])
            input_ids_left.append(input_id)
            input_masksleft.append(input_mask)
            segment_ids_left.append(segment_id)
        return input_ids_left,input_mask_left,segment_ids_left

    def get_input_matirx(self，right_tokens):
        input_ids_right, input_masks_right, segment_ids_right = [], [], []
        for i in range(len(left_tokens)):
            input_id = self.get_ids(right_tokens[i],tokenizer)
            input_mask = self.get_masks(right_tokens[i])
            segment_id = self.get_segments(right_tokens[i])
            input_ids_right.append(input_id)
            input_masks_right.append(input_mask)
            segment_ids_right.append(segment_id)
        return input_ids_right,input_mask_right,segment_ids_right

    
    def get_tokens(self，left,right,label):
        left_tokens = [["[CLS]"] + tokenizer.tokenize(question) + ["[SEP]"] for question in left]
        right_tokens = [["[CLS]"] + tokenizer.tokenize(question) + ["[SEP]"] for question in right]
        # delete the sentences that are longer than 100 tokens to fit in GPU memory
        pop_ls=[]
        for i in range(len(left_tokens)):
            if len(left_tokens[i])>100 or len(right_tokens[i])>100:
                pop_ls.append(i)
        for index in pop_ls[ : : -1]:
            left_tokens.pop(index)
            right_tokens.pop(index)
            label.pop(index)
        assert len(left_tokens) == len(right_bert) == len(label)
        return left_tokens,right_tokens,label
    
    def build_model():
        max_seq_length = 100  # Your choice here.
        input_ids_left = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids_left")
        input_ids_right = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids_right")
        input_masks_left = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask_left")
        input_masks_right = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask_right")
        segment_ids_left = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids_left")
        segment_ids_right = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids_right")
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=True)
        pooled_output_left, sequence_output_left = bert_layer([input_ids_left, input_masks_left, segment_ids_left])
        pooled_output_right, sequence_output_right = bert_layer([input_ids_right, input_masks_right, segment_ids_right])
        y = tf.keras.layers.Concatenate()([pooled_output_left, pooled_output_right])
        
        y = tf.keras.layers.Dropout(0.2)(y)
        
        pred = tf.keras.layers.Dense(1, activation='sigmoid')(y)
        #pred = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)

        model = Model(inputs=[input_ids_left, input_masks_left, segment_ids_left,input_ids_right, input_masks_right, segment_ids_right], outputs=[pred])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, ),loss="binary_crossentropy",
          metrics=["accuracy"])
        return model
    
    def trainTestSplit():
        train_input_ids_left, test_input_ids_left, train_input_masks_left, test_input_masks_left, train_segment_ids_left, test_segment_ids_left,train_input_ids_right, test_input_ids_right, train_input_masks_right, test_input_masks_right, train_segment_ids_right, test_segment_ids_right, train_Y, test_Y = train_test_split(
        input_ids_left, input_masks_left, segment_ids_left,input_ids_right, input_masks_right, segment_ids_right, label, test_size = 0.2)
        train_input_ids_left= np.asarray(train_input_ids_left,dtype = np.int32)
        test_input_ids_left = np.asarray(test_input_ids_left,dtype = np.int32)
        train_input_masks_left = np.asarray(train_input_masks_left,dtype = np.int32)
        test_input_masks_left= np.asarray(test_input_masks_left,dtype = np.int32)
        train_segment_ids_left= np.asarray(train_segment_ids_left,dtype = np.int32)
        test_segment_ids_left= np.asarray(test_segment_ids_left,dtype = np.int32)
        train_input_ids_right= np.asarray(train_input_ids_right,dtype = np.int32)
        test_input_ids_right = np.asarray(test_input_ids_right,dtype = np.int32)
        train_input_masks_right = np.asarray(train_input_masks_right,dtype = np.int32)
        test_input_masks_right= np.asarray(test_input_masks_right,dtype = np.int32)
        train_segment_ids_right= np.asarray(train_segment_ids_right,dtype = np.int32)
        test_segment_ids_right= np.asarray(test_segment_ids_right,dtype = np.int32)
        train_Y= np.asarray(train_Y,dtype = np.int32)
        test_Y= np.asarray(test_Y,dtype = np.int32)
        return train_input_ids_left, test_input_ids_left, train_input_masks_left, test_input_masks_left, train_segment_ids_left, test_segment_ids_left,train_input_ids_right, test_input_ids_right, train_input_masks_right, test_input_masks_right, train_segment_ids_right, test_segment_ids_right, train_Y, test_Y






    
