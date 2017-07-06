#!/usr/bin/python
# coding=utf-8
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import data_helper
import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score
import sklearn.tree as tree

def tf_idf_model(x_train,x_test):

    #将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    count_vect = CountVectorizer(ngram_range=(1, 2), min_df=100, max_features=10000)
    X_train_counts = count_vect.fit_transform(x_train)

    print count_vect.get_feature_names()
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(X_train_counts)

    x_test = count_vect.transform(x_test)
    x_test = tfidf_transformer.transform(x_test)
    return x_train, x_test


def train():
    train_data_name = data_helper.small_sample_dir + 'data_only_command.train'
    test_data_name = data_helper.small_sample_dir + 'data_only_command.test'
    train_x, train_y = data_helper.prepare_classification_data(train_data_name)
    test_x, test_y = data_helper.prepare_classification_data(test_data_name)
    print len(train_y)
    x_train, x_test = tf_idf_model(train_x,test_x)

    #clf = MultinomialNB().fit(x_train, y_train)
    clf = tree.DecisionTreeClassifier().fit(x_train, train_y)
    predict = clf.predict(x_test)
    print len(predict)
    print len(test_y)
    print 'crosstab:{0}'.format(pd.crosstab(test_y,predict,margins=True))
    print 'precision:{0}'.format(precision_score(test_y,predict,average='macro'))
    print 'recall:{0}'.format(recall_score(test_y,predict,average='macro'))


if __name__ == "__main__":
    train()
