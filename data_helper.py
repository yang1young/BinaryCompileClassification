import os
import re
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
import clean_utils.clean_utils as cu
from keras.utils.np_utils import to_categorical

data_dir = "/home/qiaoyang/codeData/binary_code/data/raw/"
train_dir = "/home/qiaoyang/codeData/binary_code/data/train/"
test_dir = "/home/qiaoyang/codeData/binary_code/data/test/"

full_dir = "/home/qiaoyang/codeData/binary_code/data/"
small_sample_dir = "/home/qiaoyang/codeData/binary_code/data/small_sample/"

dirs = ['0/', '1/', '2/', '3/']
model_dir = '/home/qiaoyang/pythonProject/BinaryCompileClassification/models/'

def data_format(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        tag = file.split("_")[3]
        dir = ''
        if(tag=='O0'):
            dir = dirs[0]
        elif(tag=='O1'):
            dir = dirs[1]
        elif (tag == 'O2'):
            dir = dirs[2]
        elif (tag == 'O3'):
            dir = dirs[3]
        if(dir!=''):
            flag = random.random()
            if(flag<0.2):
                temp_path = test_dir
            else:
                temp_path = train_dir
            temp_file = open(temp_path+ dir + file, 'w')
            temp_file.write(open(data_dir+file,'r').read())
            temp_file.close()


def get_sample(dir,code_file_name,need_replace_number,only_command,sample_percent):
    code_train_handler_sample = open(dir + code_file_name + '.train', 'w')
    code_dev_handler_sample = open(dir + code_file_name + '.dev', 'w')
    code_test_handler_sample = open(dir + code_file_name + '.test', 'w')
    for dir in dirs:
        dir = train_dir+dir
        files = os.listdir(dir)
        if(len(files)==0):
            continue
        tag = dir.split('/')[-2]
        print tag
        for file in files:
            keep = random.random()
            if(keep<sample_percent):
                flag = random.random()
                if (flag < 0.2):
                    handler = code_test_handler_sample
                elif(flag>0.2 and flag<0.3):
                    handler = code_dev_handler_sample
                else:
                    handler = code_train_handler_sample
                codes = open(dir+file,'r').readlines()
                for code in codes:
                    code = cu.clean(code,need_replace_number)
                    if(only_command):
                        temps = code.split('$')
                        new_temps =[]
                        for temp in temps:
                            temp = temp.strip()
                            new_temps.append(temp.split(' ')[0])
                        code = ' '.join(new_temps)
                    handler.write(tag+'@'+code+'\n')
    code_train_handler_sample.close()
    code_test_handler_sample.close()
    code_dev_handler_sample.close()


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


def prepare_classification_data(data_path):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['Code', 'Tag']
    df.columns = selected
    texts = df[selected[1]].tolist()
    #texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]].tolist()
    return texts,labels

#def validate(y,y_predict):

def prepare_dl_data(data_path):

    df = pd.read_csv(data_path, sep='@', header=None, encoding='utf8', engine='python')
    selected = ['Code', 'Tag']
    df.columns = selected
    texts = df[selected[1]].tolist()
    texts = [s.encode('utf-8') for s in texts]
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
    sorted_voc = [wc[0] for wc in common_list][:max_word-3]
    word_picked = ['<unknown>', '<pad>']
    word_picked.extend(sorted_voc)
    word_index = dict()
    for word,index in zip(word_picked,range(max_word)):
        word_index[word] = index+1
    save_obj(word_index,model_dir,voc_name)
    return word_index



if __name__ == "__main__":
    #data_format(data_dir)
    #get_sample(small_sample_dir,'data',False,False,0.1)
    # train_data_name = output_dir+'data.train'
    # test_data_name = output_dir+'data.test'
    train_x,train_y = prepare_classification_data(small_sample_dir+'data.train')
    # test_x,test_y = prepare_classification_data(test_data_name)
    #prepare_dl_data(small_sample_dir+'data.train')
    get_tokenizer(train_x,10000,'')
