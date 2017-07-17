import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score

from keras_cnn_rnn import data_helper

max_features = 10000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
MAX_SENT_LENGTH = 100
NUM_CLASS = 2
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 200
MAX_EPOCH = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


def model_structure():

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    # model = Sequential()
    # model.add(Embedding(input_dim=MAX_NB_WORDS+2, output_dim=EMBEDDING_DIM, input_length=MAX_SENT_LENGTH,trainable = True))
    # model.add(Dropout(0.5))
    # initial = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    # #model.add(LSTM(200,kernel_initializer =initial,dropout=0.8,return_sequences=True).supports_masking)
    # layer1 = LSTM(100,kernel_initializer =initial,dropout=0.8)
    # layer1.supports_masking = True
    # model.add(layer1)
    # layer2 = Dense(NUM_CLASS, activation='softmax')
    # layer2.supports_masking = True
    # model.add(layer2)
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


def train(x_train, y_train):
    model = model_structure()
    #optimizer =keras.optimizers.Adagrad(lr=0.3, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model fitting - Hierachical LSTM")
    print model.summary()
    # checkpoint
    filepath = data_helper.model_dir + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=MAX_EPOCH, batch_size=80, callbacks=callbacks_list)
    return model

def reload_model(model_name):
    model = model_structure()
    # load weights
    model.load_weights(data_helper.model_dir + model_name)
    # Compile model (required to make predictions)
    optimizer = keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    print("Created model and loaded weights from file")
    return model



if __name__ == "__main__":

    model = train(x_train,y_train)
    #model = reload_model('weights-improvement-00-0.21.hdf5')
    eval_model(model,x_test,y_test)
