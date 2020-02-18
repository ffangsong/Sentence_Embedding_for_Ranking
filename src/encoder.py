import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from tensorflow.keras.models import load_model
from bert.tokenization import bert_tokenization
MAX_SEQ_LENGTH = 100
BERT_VOCAB = '/docs/uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = '/docs/uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = '/docs/uncased_L-12_H-768_A-12/bert_config.json'



def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def get_tokens(left):
    left_tokens = [["[CLS]"] + tokenizer.tokenize(question) + ["[SEP]"] for question in left]
     return left_tokens

def get_input_matrix(tokens):
    input_ids, input_masks, segment_ids = [], [], []
    for i in range(len(tokens)):
        input_id = get_ids(tokens[i])
        input_mask = get_masks(tokens[i])
        segment_id = get_segments(tokens[i])
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return input_ids, input_masks, segment_ids

def encode( ):
    tokenizer = bert_tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)
    inferModel = load_model('docs／saved_model.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
    embedding_model = Model(inputs= inferModel.inputs, outputs = inferModel.layers[6].output[0])
    q_df = pd.read_csv('./data/doc_repository.txt').dropna()
    q_df.columns = ['question']
    uni_q = q_df['question'].unique().tolist()
    left_tokens = get_tokens(uni_q)
    right = [' ' for i in range(len(uni_q))]
    right_tokens = get_tokens(right)
    input_ids_left, input_masks_left, segment_ids_left = get_input_matrix(left_tokens)
    input_ids_right, input_masks_right, segment_ids_right= get_input_matrix(right_tokens)
    q_embedding = embedding_model.predict([input_ids_left, input_masks_left, segment_ids_left,input_ids_right, input_masks_right, segment_ids_right])
    embedding_df = pd.DataFrame(q_embedding)
    question_embedding_df = q_df.merge(embedding_df,left_index = True,right_index = True)
    question_embedding_df.to_csv('./docs/question_embedding.csv',index =False, header = False)
    return

if __name__ == '__main__':
    encode()
