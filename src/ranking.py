import os
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import json



def ranking():
        with open('./docs/index_sen_20000.json','r') as file:
                index_sen = json.load(file)
        
        f = 768

        t = AnnoyIndex(768,'euclidean')
        t.load('./docs/sentence_embedding_20000_200.ann')

        query = input('Enter your sentence here')


        print('Top 3 Related  Questions:')
        encoder = SentenceTransformer('bert-base-nli-mean-tokens')
        query_vector = encoder.encode([query])
        query_vector_ls = query_vector[0].tolist()
        output = t.get_nns_by_vector(query_vector_ls,3,search_k =3, include_distances=True)
        output_sen1 = index_sen[str(output[0][0])]
        print(output_sen1)
        output_sen2 = index_sen[str(output[0][1])]
        print(output_sen2)
        output_sen3 = index_sen[str(output[0][2])]
        print(output_sen3)
                
 if __name__ == '__main__':
        ranking()                                                                                                          

                                                                                                                
