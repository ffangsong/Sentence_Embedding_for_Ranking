import pandas as pd
import json
import csv
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex


    
def encode():
    q_df = pd.read_csv('./data/doc_repository.txt', delimiter = ' ').dropna()
    q_df.columns = ['question']
    uni_q = q_df['question'].unique().tolist()
    encoder = SentenceTransformer('bert-base-nli-mean-tokens')
    q_embedding = encoder.encode(uni_q)
    embedding_df = pd.DataFrame(q_embedding)
    question_embedding_df = q_df.merge(embedding_df,left_index = True,right_index = True)
    question_embedding_df.to_csv('./docs/question_embedding.csv',index =False, header = False)
    return

def build_index():
    f = 768
    t = AnnoyIndex(768,'euclidean')
    index = -1
    index_sen={}
    sen_index = {}
    with open('./docs/question_embedding.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            index += 1
            sentence = line[0]
            embedding=line[1:]
            embedding_vec = [float(weight) for weight in embedding]
            t.add_item(index,embedding_vec)
            index_sen.setdefault(index,sentence)
            sen_index.setdefault(sentence,index)
    t.build(200)
    t.save('./docs/index.ann')
    with open('./docs/index_sen','w') as fp:
        json.dump(index_sen,fp)
    
    return

if __name__ =='__main__':
    encode()
    build_index()

    
