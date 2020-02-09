import w2v_LSTM

import keras
# max_seq_length = 130

def main(max_seq_length =130, batch_size,n_epoch):
    w2v= w2v_LSTM.word2vec_LSTM('~/Documents/Insight/test_python.csv','~/Documents/Insight/Project/word2vec/GoogleNews-vectors-negative300.bin.gz',300)
    df = w2v.load_dataset()
    w2v_model = w2v.w2v_model()
    df = w2v.get_indicies(df,w2v_model)
    embeddings = model.creat_embedding_matrix(w2v)
    model = build_the_model(max_seq_length,embeddings)
    X_train, X_test, y_train, y_test = trainTestSplit(df)
    saver = keras.callbacks.ModelCheckpoint('/s3mnt/model_checkpoint/word2vec_LSTM/weights.{epoch:02d}-{val_acc:.4f}.hdf5')
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_acc')
    history = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_test['left'], X_test['right']], y_test),callbacks = [saver,eartlyStopping ])    
    return model, history

if name == '__main__':
    main()

