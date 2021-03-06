import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers import Embedding, LSTM, TimeDistributed
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score, accuracy_score

import clean_utils.clean_utils as cu
import data_helper

MAX_SENT_LENGTH = 150
MAX_SENTS = 10
NUM_CLASS = 104
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
MAX_EPOCH = 1

def data_transfer(word_index,x,y):
    data = np.zeros((len(x), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    for i, sentences in enumerate(x):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = cu._WORD_SPLIT.split(sent)
                wordTokens = cu.remove_blank(wordTokens)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH:
                        if(word not in word_index):
                            data[i, j, k] = word_index['<unknown>']
                        else:
                            data[i, j, k] = word_index[word]
                    k = k + 1
    labels = to_categorical(np.asarray(y),num_classes=NUM_CLASS)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data,labels

def model_structure():
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    # embedded_sequences = embedding_layer(sentence_input)
    embedded_sequences = Embedding(input_dim=MAX_NB_WORDS+1,output_dim=EMBEDDING_DIM, input_length=MAX_SENT_LENGTH,trainable=True,mask_zero=True)(
        sentence_input)
    initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    l_lstm = LSTM(200,kernel_initializer =initial,dropout=0.5)(embedded_sequences)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='float32')
    #review_input = Masking(mask_value=12345,input_shape=(MAX_SENTS, MAX_SENT_LENGTH))(temp_input)
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = LSTM(200,kernel_initializer =initial,dropout=0.5)(review_encoder)
    preds = Dense(NUM_CLASS, activation='softmax')(l_lstm_sent)
    model = Model(review_input, preds)
    return model


def eval_model(model,x,y):

    predict = model.predict(x)
    print predict
    assert len(predict)==len(y)
    y_predict = []
    y_real = []
    for p, r in zip(predict, y):
        y_predict.append(np.argmax(p))
        y_real.append(np.argmax(r))
    y_predict = np.asarray(y_predict)
    y_real = np.asarray(y_real)

    print 'crosstab:{0}'.format(pd.crosstab(y_real, y_predict, margins=True))
    print 'precision:{0}'.format(precision_score(y_real, y_predict, average='macro'))
    print 'recall:{0}'.format(recall_score(y_real, y_predict, average='macro'))
    print 'accuracy:{0}'.format(accuracy_score(y_real, y_predict))


def train(x_train, y_train,x_val, y_val,model_path):
    model = model_structure()
    #optimizer =keras.optimizers.Adagrad(lr=0.3, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model fitting - Hierachical LSTM")
    print model.summary()
    # checkpoint
    filepath = model_path+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history =model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)
    print(history.history)
    data_helper.save_obj(history.history, model_path, "train.log")
    return model

def reload_model(model_name):
    model = model_structure()
    # load weights
    model.load_weights(model_name)
    # Compile model (required to make predictions)
    #optimizer = keras.optimizers.Adagrad(lr=0.3, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("Created model and loaded weights from file")
    return model



if __name__ == "__main__":
    path = ""
    data_path = path+""
    train_path = data_path+"train.txt"
    dev_path = data_path+"dev.txt"
    test_path = data_path+"test.txt"
    model_path = path + 'model/'

    is_bytecode = False
    train_texts, train_blocks,train_labels = data_helper.prepare_dl_data(train_path, is_bytecode)
    _,dev_blocks, dev_labels = data_helper.prepare_dl_data(dev_path, is_bytecode)
    _,test_blocks,test_labels = data_helper.prepare_dl_data(test_path, is_bytecode)
    #_, test_blocks_d,test_labels_d = data_helper.prepare_dl_data(test_path_d, is_bytecode)
    print train_texts
    word_index = data_helper.get_tokenizer(train_texts, MAX_NB_WORDS, 'voca')

    print('Total %s unique tokens.' % len(word_index))

    x_train, y_train = data_transfer(word_index, train_blocks, train_labels)
    x_val, y_val = data_transfer(word_index, dev_blocks, dev_labels)
    x_test, y_test = data_transfer(word_index, test_blocks, test_labels)
    #x_test_d, y_test_d = data_transfer(word_index, test_blocks_d, test_labels_d)

    model = train(x_train, y_train, x_val, y_val, model_path)
    # model = reload_model('weights-improvement-00-0.21.hdf5')
    eval_model(model, x_test, y_test)
    #eval_model(model, x_test_d, y_test_d)
