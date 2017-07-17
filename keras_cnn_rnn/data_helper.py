import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
import clean_utils.clean_utils as cu
from keras.utils.np_utils import to_categorical

data_path = "/home/qiaoyang/codeData/binary_code/data/small_sample/"
model_dir = '/home/qiaoyang/pythonProject/BinaryCompileClassification/models/'

# process labels
def label_to_categorical(labels, need_to_categorical):
    label = sorted(list(set(labels)))
    num_labels = len(label)
    print('label total count is: ' + str(num_labels))
    label_indict = range(num_labels)
    labels_index = dict(zip(label, label_indict))
    labels = [labels_index[y] for y in labels]
    if (need_to_categorical):
        labels = to_categorical(np.asarray(labels))
    print('Shape of label tensor:', labels.shape)
    return labels, label


def prepare_classification_data(data_path,is_bytecode):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['tag', 'assemble','byte']
    df.columns = selected
    code_index =1
    if(is_bytecode):
        code_index =2
    texts = df[selected[code_index]].tolist()
    #texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]].tolist()
    return texts,labels


def prepare_dl_data(data_path,is_bytecode):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['tag', 'assemble', 'byte']
    df.columns = selected
    code_index = 1
    if (is_bytecode):
        code_index = 2
    texts = df[selected[code_index]].tolist()
    # texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]].tolist()

    blocks = []
    for text in texts:
        codes = str(text).split('$')
        blocks.append(codes[:-1])
    return texts,blocks,labels


def save_obj(obj,path, name):
    with open(path+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path,name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_tokenizer(all_text,max_word,voc_name):

    texts = []
    for text in all_text:
        temp = cu.remove_blank(cu._WORD_SPLIT.split(text))
        texts.extend(temp)
    counts = Counter(texts)
    common_list = counts.most_common()
    common_list.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in common_list]
    word_picked = ['<unknown>']
    word_picked.extend(sorted_voc)
    if(len(word_picked)>max_word):
        word_picked = word_picked[:max_word]
    word_index = dict()
    for word,index in zip(word_picked,range(max_word)):
        word_index[word] = index+1
    save_obj(word_index,model_dir,voc_name)
    print "unknown word index is "+str(word_index.get('<unknown>'))
    print "Nuber of unique token is "+str(len(word_index))
    return word_index



if __name__ == "__main__":
    print ''
