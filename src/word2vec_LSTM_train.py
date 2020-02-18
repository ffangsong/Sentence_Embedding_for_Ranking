from word2vec_LSTM import word2vec_LSTM
import re
import keras
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 130

def main(MAX_SEQ_LENGTH, BATCH_SIZE,n_EPOCH):
    w2v = word2vec_LSTM()
        
    df = w2v.load_dataset()

    df = w2v.get_indicies(df)
    embeddings = w2v.creat_embedding_matrix()
    model = w2v.build_model(embeddings)
    X_train, X_test, y_train, y_test = w2v.trainTestSplit(df)
    X_train,X_test = w2v.pad_sequence(X_train,X_test)
    saver = keras.callbacks.ModelCheckpoint('./docs/model_checkpoint/word2vec_LSTM/weights.{epoch:02d}-{val_acc:.4f}.hdf5')
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc')
    history = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_test['left'], X_test['right']], y_test), callbacks=[saver, earlyStopping])
    return history

if __name__=='__main__':
    main(MAX_SEQ_LENGTH=130, BATCH_SIZE=64, n_EPOCH=10)
