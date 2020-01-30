import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from gensim.models import KeyedVectors
data_file_path = '~/Documents/Insight/test_python.csv'
embedding_file_paht ='~/Documents/Insight/Project/word2vec/GoogleNews-vectors-negative300.bin.gz' 
embedding_dim =300
max_seq_length = 130

class word2vec_LSTM():
    def __init__(self,data_file_path,embedding_file_path,embedding_dim):
        self.embedding_file_path = embedding_file_path
        self.data_file_path = data_file_path
        self.embedding_dim = embedding_dim
        self.max_seq_length = 0
        self.vocabulary = {}
        self.inverse_vocabulary=inverse_vocabulary = ['<unk']
        #self.w2v_model = KeyedVectors.load_word2vec_format(self.embedding_file_path, binary = True)
        
    def load_dataset(self):
        df = pd.read_csv(self.data_file_path,header = 0, sep = ',').dropna()
        return df 

    # convert sentence to list of words and lemmatize the words  
    def text_to_word(self,text):
        lemmatizer = WordNetLemmatizer()
        text = str(text)
        text = str(text)
        # split text into individual words
        words = word_tokenize(text)
        # Lemmatize
        words = list(map(lambda x : lemmatizer.lemmatize(x,'v'),words))
        return words
    
    def w2v_model(self ):
         w2v_model = KeyedVectors.load_word2vec_format(self.embedding_file_path, binary = True)
         return w2v_model
    
    # Replace questions with of lists of indices, include stopwords if they have embedding
    def get_indicies(self,df,w2v_model):
        stop_words = set(stopwords.words('english'))
        questions_cols = ['question1','question2']
        for index,row in df.iterrows():
            for question in questions_cols:
                q2n = []
                for word in self.text_to_word(row[question]):
                    if word in stop_words and word not in self.w2v_model.vocab:
                        continue
                    if word not in self.vocabulary:
                        vocabulary[word] = len(self.inverse_vocabulary)
                        q2n.append(len(self.inverse_vocabulary))
                        self.inverse_vocabulary.append(word)
                    else:
                        q2n.append(self.vocabulary[word])
                df.set_value(index,question,q2n) 
        return df

    # create the embedding matrix

    def creat_embedding_matrix(self,w2v_model):
        embeddings = 1* np.random.randn(len(vocabulary)+1,self.embedding_dim)
        embeddings[0]=0
        for word, index in vocabulary.items():
            if word in w2v_model.vocab:
                embeddings[index] = self.w2v_model.word_vec(word)
        return embeddings
    
    def trainTestSplit(self,df):
        X = df[['question1','question2']]
        y = df['is_duplicate']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
        X_train = {'left': X_train.question1, 'right': X_train.question2}
        X_test = {'left': X_test.question1, 'right': X_test.question2}
        y_train = y_train.values
        y_test = y_test.values
        return X_train, X_test,y_train,y_test

    def build_the_model(self,max_seq_length,embeddings):
        # The Input layer
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')
        embedding_layer = Embedding(len(embeddings), self.embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        model = Model(input=[left_input, right_input], output =[malstm_distance])

        # Adadelta optimizer, with gradient clipping by norm
        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        return model
     


