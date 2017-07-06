#!/usr/bin/python
# coding=utf-8
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import data_helper
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score
import sklearn.tree as tree
import clean_utils.clean_utils as cu
import numpy as np
def tf_idf_model(x_train,x_test):

    #将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vect = CountVectorizer(ngram_range=(1, 1), min_df=100, max_features=10000)
    X_train_counts = count_vect.fit_transform(x_train)

    print count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)

    x_test = count_vect.transform(x_test)
    x_test = tfidf_transformer.transform(x_test)
    return x_train, x_test,count_vect,tfidf_transformer


def train():
    train_data_name = data_helper.small_sample_dir + 'data.train'
    test_data_name = data_helper.small_sample_dir + 'data.test'
    train_x, train_y = data_helper.prepare_classification_data(train_data_name)
    test_x, test_y = data_helper.prepare_classification_data(test_data_name)
    print len(train_y)
    x_train, x_test,count_vect,tfidf_transformer = tf_idf_model(train_x,test_x)

    #clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(x_train, train_y)
    predict = clf.predict(x_test)
    print len(predict)
    print len(test_y)
    print 'crosstab:{0}'.format(pd.crosstab(test_y,predict,margins=True))
    print 'precision:{0}'.format(precision_score(test_y,predict,average='macro'))
    print 'recall:{0}'.format(recall_score(test_y,predict,average='macro'))
    return clf,count_vect,tfidf_transformer


def ensamble_test(model,count_vect, tfidf_transformer,need_replace_number):
    y = []
    y_stack = []
    predict = []
    predict_stack = []
    for dir in data_helper.dirs:
        dir = data_helper.test_dir+dir
        files = os.listdir(dir)
        if(len(files)==0):
            continue
        tag = dir.split('/')[-2]
        print tag
        for file in files:
            codes = open(dir + file, 'r').readlines()
            y_stack.append(tag)
            temp_y_pred = []
            for code in codes:
                y.append(tag)
                code = cu.clean(code, need_replace_number)
                x = tfidf_transformer.transform(count_vect.transform([code]))
                pred = model.predict(x)[0]
                predict.append(str(pred))
                temp_y_pred.append(pred)
            y_stack_predict = max(temp_y_pred,key=temp_y_pred.count)
            predict_stack.append(str(y_stack_predict))
    print len(y)
    print len(y_stack)
    print len(predict)
    print len(predict_stack)
    y = np.asarray(y)
    y_stack = np.asarray(y_stack)
    predict = np.asarray(predict)
    predict_stack = np.asarray(predict_stack)
    print 'crosstab:{0}'.format(pd.crosstab(y, predict, margins=True))
    print 'crosstab:{0}'.format(pd.crosstab(y_stack,predict_stack,margins=True))
    print 'precision:{0}'.format(precision_score(y, predict, average='macro'))
    print 'recall:{0}'.format(recall_score(y, predict, average='macro'))
    print '*********************************'
    print 'precision:{0}'.format(precision_score(y_stack,predict_stack,average='macro'))
    print 'recall:{0}'.format(recall_score(y_stack,predict_stack,average='macro'))





if __name__ == "__main__":
     clf, count_vect, tfidf_transformer = train()
     ensamble_test(clf, count_vect, tfidf_transformer,False)
    # x = np.asarray([1,2,3,4,5])
    # y = np.asarray([1,2,3,4,5])
    # print 'precision:{0}'.format(precision_score(x,y, average='macro'))


