import w2v_LSTM
max_seq_length = 130
def train_test_split(df):
    X = df[['question1','question2']]
    y = df['is_duplicate']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
    X_train = {'left': X_train.question1, 'right': X_train.question2}
    X_test = {'left': X_test.question1, 'right': X_test.question2}
    y_train = y_train.values
    y_test = y_test.values
    return X_train, X_test,y_train,y_test
    
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

