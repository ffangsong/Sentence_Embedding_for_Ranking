import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')

def load_dataset():
    data_path = '~/Documents/Insight/Text_Similarity/train.csv'
    data = pd.read_csv(data_path)
    return data
def text_to_word(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text)
    # split text into individual words
    words = word_tokenize(text)
    #lemmatize
    words = list(map(lambda x: lemmatizer.lemmatize(x,'v'),words))
    words = [word for word in words if word not in stop_words]
    return words
def text_preprocessing(data):
    data['q1tokens'] = data['question1'].apply(text_to_word)
    data['q2tokens']=data['question2'].apply(text_to_word)
    return data
def main():
    data = load_dataset()
    data = text_preprocessing(data)
    return data
if __name__ == '__main__':
    main()

