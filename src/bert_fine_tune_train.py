import bert_fine_tune
from bert_fine_tune import bert_fine_tune
import tensorflow as tf
import time
#filepath = '/s3mnt/model_checkpoint/bert_siamese/'
def main():
    bert = bert_fine_tune()
    left, right, label = bert.load_dataset()
    left_tokens, right_tokens, label = bert.get_tokens(left, right, label)
    input_ids_right, input_masks_right, segment_ids_right = bert.get_input_matrix(right_tokens)
    input_ids_left, input_masks_left, segment_ids_left = bert.get_input_matrix(left_tokens)
    train_input_ids_left, test_input_ids_left, train_input_masks_left, test_input_masks_left, \
    train_segment_ids_left, test_segment_ids_left,train_input_ids_right, test_input_ids_right, \
train_input_masks_right, test_input_masks_right, train_segment_ids_right, \
	test_segment_ids_right, train_Y, test_Y=bert.trainTestSplit(input_ids_left,
input_masks_left,segment_ids_left,input_ids_right,input_masks_right,segment_ids_right,label)
    saver = tf.keras.callbacks.ModelCheckpoint("/s3mnt/model_checkpoint/bert_siamese/weights.{epoch:02d}-{val_acc:.4f}.hdf5",save_best_only =False)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc')
    model = bert.build_model()
    #training_start_time = time()
    history = model.fit([train_input_ids_left, train_input_masks_left,train_segment_ids_left,train_input_ids_right, train_input_masks_right,train_segment_ids_right], train_Y, validation_data=([test_input_ids_left,test_input_masks_left,test_segment_ids_left,test_input_ids_right,test_input_masks_right, test_segment_ids_right], test_Y), batch_size=8, epochs=2, callbacks=[saver, early_stopping])
    return history
if __name__ == '__main__':
    main()
