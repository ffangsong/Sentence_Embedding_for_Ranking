import itertools
import re
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.optimizers import Adadelta
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

DATA_FILE_PATH = './data/train_data.txt'
EMBEDDING_FILE_PATH = './docs/pretrained/GoogleNews-vectors-negative300.bin.gz'
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 130
GRADIENT_CLIPPING_NORM = 1.25


class word2vec_LSTM():
    def __init__(self):
        self.MAX_SEQ_LENGTH = 0
        self.vocabulary = {}
        self.inverse_vocabulary = inverse_vocabulary = ['<unk']
        self.w2v_model = KeyedVectors.load_word2vec_format(
            EMBEDDING_FILE_PATH,
            binary=True)

    def load_dataset(self):
        ''' load_dataset '''
        df = pd.read_csv(DATA_FILE_PATH, header=0, sep=' ').dropna()
        return df

    def text_to_word(self, text):
        ''' clean_text, and lemmatize the words'''
        text = str(text)
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", 'what is', text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
        # Remove punctuation from text
        text = ''.join([c for c in text if c not in punctuation]).lower()
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        text = str(text)
    
        # split text into individual words''
        words = word_tokenize(text)
        # Lemmatize
        words = list(map(lambda x: lemmatizer.lemmatize(x, 'v'), words))
        return words

    def get_indicies(self, df):
        '''Replace questions with of lists of indices,
        include stopwords if they have embedding'''
        stop_words = set(stopwords.words('english'))
        questions_cols = ['question1', 'question2']
        for index, row in df.iterrows():
            for question in questions_cols:
                q2n = []
                for word in self.text_to_word(row[question]):
                    if word in stop_words and word not in self.w2v_model.vocab:
                        continue
                    if word not in self.vocabulary:
                        self.vocabulary[word] = len(self.inverse_vocabulary)
                        q2n.append(len(self.inverse_vocabulary))
                        self.inverse_vocabulary.append(word)
                    else:
                        q2n.append(self.vocabulary[word])
                df.set_value(index, question, q2n)
        return df

    def creat_embedding_matrix(self):
        '''create the embedding matrix'''
        embeddings = 1 * np.random.randn(len(self.vocabulary) + 1, EMBEDDING_DIM)
        embeddings[0] = 0
        for word, index in self.vocabulary.items():
            if word in self.w2v_model.vocab:
                embeddings[index] = self.w2v_model.word_vec(word)
        return embeddings

    def trainTestSplit(self, df):
        X = df[['question1', ' question2']]
        y = df['is_duplicate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = {'left': X_train.question1, 'right': X_train.question2}
        X_test = {'left': X_test.question1, 'right': X_test.question2}
        y_train = y_train.values
        y_test = y_test.values
        return X_train, X_test, y_train, y_test

    def pad_sequence(self, X_train, X_test):
        '''pad the sequence'''
        for dataset, side in itertools.product([X_train, X_test], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=MAX_SEQ_LENGTH)
        return X_train, X_test

    def build_model(self, embeddings):
        def exponent_neg_manhattan_distance(left, right):
            return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

        # The Input layer
        left_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
        right_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(embeddings), EMBEDDING_DIM, weights=[embeddings], input_length=MAX_SEQ_LENGTH,
                                    trainable=False)

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(50)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        model = Model(input=[left_input, right_input], output=[malstm_distance])

        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=GRADIENT_CLIPPING_NORM)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        return model
