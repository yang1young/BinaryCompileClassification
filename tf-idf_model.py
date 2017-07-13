#!/usr/bin/python
# coding=utf-8
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import data_helper
import random
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import sklearn.tree as tree
import clean_utils.clean_utils as cu
import numpy as np


def tf_idf_model(x_train):
    # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vect = CountVectorizer(ngram_range=(1, 1), min_df=100, max_features=10000)
    X_train_counts = count_vect.fit_transform(x_train)
    print count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)
    return x_train, count_vect, tfidf_transformer


def eval_model(real, predict, name):
    assert len(predict) == len(real)
    print name
    print 'crosstab:{0}'.format(pd.crosstab(real, predict, margins=True))
    print 'precision:{0}'.format(precision_score(real, predict, average='macro'))
    print 'recall:{0}'.format(recall_score(real, predict, average='macro'))
    print 'accuracy:{0}'.format(accuracy_score(real, predict))


def train(train_path, test_path, test_path_duplicate, is_bytecode):
    train_x, train_y = data_helper.prepare_classification_data(train_path, is_bytecode)
    test_x, test_y = data_helper.prepare_classification_data(test_path, is_bytecode)
    test_x_d, test_y_d = data_helper.prepare_classification_data(test_path_duplicate, is_bytecode)
    print len(train_y)
    x_train, count_vect, tfidf_transformer = tf_idf_model(train_x)

    x_test = count_vect.transform(test_x)
    x_test = tfidf_transformer.transform(x_test)
    x_test_d = count_vect.transform(test_x_d)
    x_test_d = tfidf_transformer.transform(x_test_d)

    # clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(x_train, train_y)

    predict_distict = clf.predict(x_test)
    eval_model(test_y, predict_distict, "distinct data evaluation")

    predict_duplicate = clf.predict(x_test_d)
    eval_model(test_y_d, predict_duplicate, "contains duplicated data evaluation")

    return clf, count_vect, tfidf_transformer


def ensamble_test(data_path, model, sample_percent,count_vect, tfidf_transformer, is_bytecode):
    y = []
    predict = []
    predict_stack = []

    files = os.listdir(data_path)
    for file in files:
        keep = random.random()
        if (keep < sample_percent):
            tag = file.split("_")[3]
            y.append(str(tag))
            codes = open(data_path+file, 'r').readlines()
            x = []
            for code in codes:
                if(is_bytecode):
                    t = code.split('@')[1]
                    x.append(cu.bytecode_clean(t))
                else:
                    t = code.split('@')[0]
                    x.append(cu.assemble_clean(t, False))
            x = tfidf_transformer.transform(count_vect.transform(x))
            pred = model.predict(x)

            predict.append(random.choice(pred))
            predict_stack.append(max(pred, key=pred.count))
    y = np.asarray(y)
    predict = np.asarray(predict)
    predict_stack = np.asarray(predict_stack)
    eval_model(y, predict, "random choose a label from block as function result")
    eval_model(y, predict_stack, "block voting result")


if __name__ == "__main__":
    data_path = ""
    train_path = ""
    test_path = ""
    test_path_duplicate = ""
    is_bytecode = True
    clf, count_vect, tfidf_transformer = train(train_path, test_path, test_path_duplicate, is_bytecode)
    ensamble_test(data_path, clf, 0.2, count_vect, tfidf_transformer, is_bytecode)
