import keras
from sklearn.metrics import precision_score,recall_score
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
import clean_utils.clean_utils as cu
import data_helper
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, Sequential

MAX_SENT_LENGTH = 100
NUM_CLASS = 4
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 200
MAX_EPOCH = 10

def data_transfer(word_index,x,y):
    data = np.zeros((len(x), MAX_SENT_LENGTH), dtype='int32')
    for i, sentences in enumerate(x):
        wordTokens = cu._WORD_SPLIT.split(sentences)
        wordTokens = cu.remove_blank(wordTokens)
        k = 0
        for _, word in enumerate(wordTokens):
            if(k<MAX_SENT_LENGTH):
                if (word not in word_index):
                    data[i, k] = 1
                else:
                    data[i, k] = word_index[word]
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

    model = Sequential()
    model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SENT_LENGTH,trainable = True,mask_zero=True))
    model.add(Dropout(0.5))
    initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    model.add(LSTM(200,kernel_initializer =initial,dropout=0.8,return_sequences=True))
    model.add(LSTM(100,kernel_initializer =initial,dropout=0.8))
    model.add(Dense(NUM_CLASS, activation='softmax'))
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


def train(x_train, y_train,x_val, y_val):
    model = model_structure()
    optimizer =keras.optimizers.Adagrad(lr=0.3, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    print("model fitting - Hierachical LSTM")
    print model.summary()
    # checkpoint
    filepath = data_helper.model_dir+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)
    return model

def reload_model(model_name):
    model = model_structure()
    # load weights
    model.load_weights(data_helper.model_dir+model_name)
    # Compile model (required to make predictions)
    optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    print("Created model and loaded weights from file")
    return model



if __name__ == "__main__":
    train_texts, train_labels = data_helper.prepare_classification_data(data_helper.small_sample_dir + 'data.train')
    dev_texts, dev_labels = data_helper.prepare_classification_data(data_helper.small_sample_dir + 'data.dev')
    test_texts, test_labels = data_helper.prepare_classification_data(data_helper.small_sample_dir + 'data.test')

    # tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(train_texts)
    # tokenizer.word_index['UKNOWN'] = MAX_NB_WORDS
    # word_index = tokenizer.word_index
    word_index = data_helper.get_tokenizer(train_texts,MAX_NB_WORDS,'voca')

    print('Total %s unique tokens.' % len(word_index))

    x_train, y_train = data_transfer(word_index, train_texts, train_labels)
    x_val, y_val = data_transfer(word_index, dev_texts, dev_labels)
    x_test, y_test = data_transfer(word_index, test_texts, test_labels)

    model = train(x_train,y_train,x_val,y_val)
    #model = reload_model('weights-improvement-00-0.21.hdf5')
    eval_model(model,x_test,y_test)
