import w2v_LSTM
max_seq_length = 130
def main(max_seq_length =130):
    w2v= w2v_LSTM.word2vec_LSTM('~/Documents/Insight/test_python.csv','~/Documents/Insight/Project/word2vec/GoogleNews-vectors-negative300.bin.gz',300)
    df = w2v.load_dataset()
    w2v_model = w2v.w2v_model()
    df = w2v.get_indicies(df,w2v_model)
    embeddings = model.creat_embedding_matrix(w2v)
    model = build_the_model(max_seq_length,embeddings)
    return model

if name == '__main__':
    main()
