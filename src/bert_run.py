import bertFineTune
from bertFineTune import *
import time
#filepath = '/s3mnt/model_checkpoint/bert_siamese/'
def main():
    bertFine = bertFineTune
    left,right,label = load_dataset()
    tokenizer = createTokenizer()
    left_tokens, right_tokens = get_tokens(left,right,label,tokenizer)
    input_ids_right,input_mask_right,segment_ids_right = get_input_matrix_right(right_tokens)
    input_ids_left,input_mask_left,segment_ids_left = get_input_matrix_left(left_tokens)
    train_input_ids_left, test_input_ids_left, train_input_masks_left, test_input_masks_left, train_segment_ids_left, test_segment_ids_left,train_input_ids_right, test_input_ids_right, train_input_masks_right, test_input_masks_right, train_segment_ids_right,
    test_segment_ids_right, train_Y, test_Y = trainTestSplit(input_ids_left,input_mask_left,segment_ids_left,input_ids_right,input_mask_right,segment_ids_right,label)
    saver = tf.keras.callbacks.ModelCheckpoint("/s3mnt/model_checkpoint/bert_siamese/weights.{epoch:02d}-{val_acc:.4f}.hdf5",save_best_only =False)
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_acc')
    training_start_time = time()
    history = model.fit([train_input_ids_left, train_input_masks_left,train_segment_ids_left,train_input_ids_right, train_input_masks_right,train_segment_ids_right], train_Y, validation_data=([test_input_ids_left,test_input_masks_left,test_segment_ids_left,test_input_ids_right,test_input_masks_right,test_segment_ids_right], test_Y), batch_size=8, epochs=5,callbacks = [saver,earlystopping])

    return model
if name == '__main__':
    main()
