"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import cPickle as pkl
import pandas as pd
import tensorflow_cnn_rnn.data_helper as daf

def padding_and_generate_mask(x, y, new_x, new_y, new_mask_x,max_len):
    for i, (x, y) in enumerate(zip(x, y)):
        # whether to remove sentences with length larger than maxlen
        if len(x) <= max_len:
            new_x[i, 0:len(x)] = x
            new_mask_x[0:len(x), i] = 1

            new_y[i] = np.array(y)
        else:
            new_x[i] = (x[0:max_len])
            new_mask_x[:, i] = 1

            new_y[i] = np.array(y)

    new_set = (new_x, new_y, new_mask_x)
    del new_x, new_y
    return new_set


def load_data(filename,max_length,max_word,is_bytecode,need_replace_number):

    x_train, y_train, vocabulary, vocabulary_inv, df, labels, label_dict = daf.load_data(
        filename+ '.train', max_length, max_word, False,is_bytecode, need_replace_number)
    print y_train
    x_dev, y_dev = daf.load_dev_test_data(filename+ '.dev', max_length, vocabulary, label_dict,
                                                  False,is_bytecode, need_replace_number)
    x_test, y_test = daf.load_dev_test_data(filename + '.test', max_length, vocabulary,
                                                    label_dict, False,is_bytecode, need_replace_number)

    class_num = 4
    new_train_set_x=np.zeros([len(y_train),max_length])
    new_train_set_y=np.zeros([len(y_train),class_num])

    new_valid_set_x=np.zeros([len(y_dev),max_length])
    new_valid_set_y=np.zeros([len(y_dev),class_num])

    new_test_set_x=np.zeros([len(y_test),max_length])
    new_test_set_y=np.zeros([len(y_test),class_num])

    mask_train_x=np.zeros([max_length,len(y_train)])
    mask_test_x=np.zeros([max_length,len(y_test)])
    mask_valid_x=np.zeros([max_length,len(y_dev)])

    train_set=padding_and_generate_mask(x_train,y_train,new_train_set_x,new_train_set_y,mask_train_x,max_length)
    test_set=padding_and_generate_mask(x_test,y_test,new_test_set_x,new_test_set_y,mask_test_x,max_length)
    valid_set=padding_and_generate_mask(x_dev,y_dev,new_valid_set_x,new_valid_set_y,mask_valid_x,max_length)

    return train_set,valid_set,test_set


#return batch dataset
def batch_iter(data,batch_size):

    #get dataset and label
    x,y,mask_x=data
    x=np.array(x)
    y=np.array(y)
    data_size=len(x)
    num_batches_per_epoch=int((data_size-1)/batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_x = x[start_index:end_index]
        return_y = y[start_index:end_index]
        return_mask_x = mask_x[:,start_index:end_index]
        # if(len(return_x)<batch_size):
        #     print(len(return_x))
        #     print return_x
        #     print return_y
        #     print return_mask_x
        #     import sys
        #     sys.exit(0)
        yield (return_x,return_y,return_mask_x)


