import codecs
import os
import re
import clean_utils.clean_utils as cc

path = '/home/qiaoyang/bisheData/binary_result/'

write_to = '/home/qiaoyang/bishe/BinaryCompileClassification/data/'
tagSet = range(0,104)


def prepare_csv():
    code_train = codecs.open(write_to + 'train.txt', 'w+', 'utf8')
    code_dev = codecs.open(write_to + 'dev.txt', 'w+', 'utf8')
    code_test = codecs.open(write_to + 'test.txt', 'w+', 'utf8')
    dirs = os.listdir(path)
    dirs.sort(key=lambda x: int(x))
    for d in dirs:
        files = os.listdir(path+d)
        files.sort(key=lambda y: int(str(y).split('.')[0]))
        for f in files:
            codes = open(path+d+'/'+f,'r').read()
            blocks = codes.split('#')
            assemblys = []
            bytes = []
            for block in blocks:
                assembly = block.split('@')[0]
                byte = block.split('@')[1]
                assemblys.append(cc.assemble_clean(assembly,False,-1,False))
                bytes.append(cc.bytecode_clean(byte,-1,False))

            tag = int(d) - 1
            i = int(str(f).split('.')[0])
            print i
            if (i < 500 * 0.7):
                code_train.write(str(tag) + "@" + '#'.join(assemblys) + "@" + '#'.join(bytes) + "\n")
            elif (i < 500 * 0.8):
                code_dev.write(str(tag) + "@" + '#'.join(assemblys) + "@" + '#'.join(bytes) + "\n")
            else:
                code_test.write(str(tag) + "@" + '#'.join(assemblys) + "@" + '#'.join(bytes) + "\n")


#prepare_csv()