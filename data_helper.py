import os
import random
import numpy as np
import pandas as pd
import clean_utils.clean_utils as cu
from keras.utils.np_utils import to_categorical

data_dir = "/home/qiaoyang/codeData/binary_code/data/result/"
train_dir = "/home/qiaoyang/codeData/binary_code/data/train_file/"
test_dir = "/home/qiaoyang/codeData/binary_code/data/test_file/"
output_dir = "/home/qiaoyang/codeData/binary_code/data/"
small_sample_dir = "/home/qiaoyang/codeData/binary_code/data/small_sample/"
dirs = [output_dir + '0/', output_dir + '1/', output_dir + '2/', output_dir + '3/']


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
            temp_file = open(dir + file, 'w')
            temp_file.write(open(data_dir+file,'r').read())
            temp_file.close()


def sep_file_to_whole(code_file_name):
    code_train_handler = open(output_dir+code_file_name+'.train','w')
    code_dev_handler = open(output_dir+code_file_name+'.dev','w')
    code_test_handler = open(output_dir+code_file_name+'.test', 'w')
    for dir in dirs:
        files = os.listdir(dir)
        if(len(files)==0):
            continue
        tag = dir.split('/')[-2]
        print tag
        for file in files:
            flag = random.random()
            if (flag < 0.2):
                handler = code_test_handler
            elif(flag>0.2 and flag<0.3):
                handler = code_dev_handler
            else:
                handler = code_train_handler
            codes = open(dir+file,'r').readlines()
            for code in codes:
                code = cu.clean(code,False)
                handler.write(tag+'@'+code+'\n')
    code_train_handler.close()
    code_test_handler.close()
    code_dev_handler.close()

def get_small_sample(code_file_name,only_command):
    code_train_handler_sample = open(small_sample_dir + code_file_name + '.train', 'w')
    code_dev_handler_sample = open(small_sample_dir + code_file_name + '.dev', 'w')
    code_test_handler_sample = open(small_sample_dir + code_file_name + '.test', 'w')
    for dir in dirs:
        files = os.listdir(dir)
        if(len(files)==0):
            continue
        tag = dir.split('/')[-2]
        print tag
        for file in files:
            keep = random.random()
            if(keep<0.1):
                flag = random.random()
                if (flag < 0.2):
                    handler = code_test_handler_sample
                elif(flag>0.2 and flag<0.3):
                    handler = code_dev_handler_sample
                else:
                    handler = code_train_handler_sample
                codes = open(dir+file,'r').readlines()
                for code in codes:
                    code = cu.clean(code,False)
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
    texts = df[selected[1]]
    #texts = [s.encode('utf-8') for s in texts]
    labels = df[selected[0]]
    return texts,labels

#def validate(y,y_predict):




if __name__ == "__main__":
    #data_format(data_dir)
    #sep_file_to_whole('data')
    get_small_sample('data_only_command',True)
    # train_data_name = output_dir+'data.train'
    # test_data_name = output_dir+'data.test'
    # train_x,train_y = prepare_classification_data(train_data_name)
    # test_x,test_y = prepare_classification_data(test_data_name)

