import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, LSTM
from keras.layers import Dense, Input, Flatten,Dropout
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras.layers.normalization import BatchNormalization
import clean_utils.clean_utils as cu
import data_helper

MAX_SENT_LENGTH = 150
NUM_CLASS = 104
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
MAX_EPOCH = 1

def data_transfer(word_index,x,y):
    data = np.zeros((len(x), MAX_SENT_LENGTH), dtype='int32')
    for i, sentences in enumerate(x):
        wordTokens = cu._WORD_SPLIT.split(sentences)
        wordTokens = cu.remove_blank(wordTokens)
        k = 0
        for _, word in enumerate(wordTokens):
            if(k<MAX_SENT_LENGTH):
                if (word not in word_index):
                    data[i, k] = word_index['<unknown>']
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

def rnn_model():

    model = Sequential()
    model.add(Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS+1, input_length=MAX_SENT_LENGTH,trainable = True,mask_zero=True))
    #model.add(Masking(mask_value=0))
    #model.add(Dropout(0.5))
    initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    #model.add(LSTM(200,kernel_initializer =initial,dropout=0.8,return_sequences=True).supports_masking)
    model.add(LSTM(300,kernel_initializer =initial,dropout=0.5,return_sequences=True))
    model.add(LSTM(200,kernel_initializer =initial,dropout=0.5))
    model.add(Dense(NUM_CLASS, activation='softmax'))
    return model


def cnn_model():
    convs = []
    filter_sizes = [3, 4, 5]
    sequence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS, input_length=MAX_SENT_LENGTH,trainable = True)(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_dropout = Dropout(0.5)(l_merge)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_dropout)
    l_bn = BatchNormalization()(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_bn)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(NUM_CLASS, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    print model.summary()
    return model


from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,Embedding, Merge, LSTM
from keras.layers import Dense, Input, Flatten,Dropout,Reshape
def cnn2_model():

    filter_sizes = [3, 4, 5]
    inputs = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=MAX_NB_WORDS+1, input_length=MAX_SENT_LENGTH,trainable = True)(inputs)
    reshape = Reshape((MAX_SENT_LENGTH, EMBEDDING_DIM, 1))(embedding)

    conv_0 = Conv2D(512, filter_sizes[0], EMBEDDING_DIM, border_mode='valid', init='normal',activation='relu', dim_ordering='tf')(reshape)
    conv_1 = Conv2D(512, filter_sizes[1], EMBEDDING_DIM, border_mode='valid', init='normal',activation='relu', dim_ordering='tf')(reshape)
    conv_2 = Conv2D(512, filter_sizes[2], EMBEDDING_DIM, border_mode='valid', init='normal',activation='relu', dim_ordering='tf')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(MAX_SENT_LENGTH - filter_sizes[0] + 1, 1), strides=(1, 1), border_mode='valid',dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(MAX_SENT_LENGTH - filter_sizes[1] + 1, 1), strides=(1, 1), border_mode='valid',dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(MAX_SENT_LENGTH - filter_sizes[2] + 1, 1), strides=(1, 1), border_mode='valid',dim_ordering='tf')(conv_2)

    merged_tensor = Merge(mode='concat', concat_axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(merged_tensor)
    # reshape = Reshape((3*num_filters,))(merged_tensor)
    dropout = Dropout(0.5)(flatten)

    output = Dense(NUM_CLASS, activation='softmax')(dropout)
    # this creates a model that includes
    model = Model(input=inputs, output=output)
    print model.summary()
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
    model = rnn_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model fitting - Hierachical LSTM")
    print model.summary()
    # checkpoint
    filepath = model_path+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)
    print(history.history)
    data_helper.save_obj(history.history, model_path, "train.log")
    return model

def reload_model(model_path,model_name):
    model = rnn_model()
    # load weights
    model.load_weights(model_path + model_name)
    # Compile model (required to make predictions)
    #optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("Created model and loaded weights from file")
    return model



if __name__ == "__main__":
    path = "/home/qiaoyang/bishe/BinaryCompileClassification/"
    data_path = path+"data/"
    train_path = data_path+"train.txt"
    dev_path = data_path+"dev.txt"
    test_path = data_path+"test.txt"
    model_path = path+'model/'

    is_bytecode = False
    train_texts, train_labels = data_helper.prepare_classification_data(train_path, is_bytecode)
    dev_texts, dev_labels = data_helper.prepare_classification_data(dev_path, is_bytecode)
    test_texts, test_labels = data_helper.prepare_classification_data(test_path, is_bytecode)
    #test_texts_d, test_labels_d = data_helper.prepare_classification_data(test_path_d, is_bytecode)

    # tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(train_texts)
    # tokenizer.word_index['UKNOWN'] = MAX_NB_WORDS
    # word_index = tokenizer.word_index
    word_index = data_helper.get_tokenizer(train_texts, MAX_NB_WORDS, model_path)

    print('Total %s unique tokens.' % len(word_index))

    x_train, y_train = data_transfer(word_index, train_texts, train_labels)
    x_val, y_val = data_transfer(word_index, dev_texts, dev_labels)
    x_test, y_test = data_transfer(word_index, test_texts, test_labels)
    #x_test_d, y_test_d = data_transfer(word_index, test_texts_d, test_labels_d)

    model = train(x_train,y_train,x_val,y_val,model_path)
    #model = reload_model(model_path,'weights-improvement-05-0.51.hdf5')
    eval_model(model,x_test,y_test)
    #eval_model(model,x_test_d,y_test_d)