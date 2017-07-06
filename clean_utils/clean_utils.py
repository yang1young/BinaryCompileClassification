#!/usr/bin/python
# coding=utf-8
import re
import sys
import mistune
reload(sys)
sys.setdefaultencoding('utf-8')

_WORD_SPLIT = re.compile(b"([,+\-&!%'_?|=\s/\*^<>$@\[\](){}#;])")
SPLIT_CHARS = [',','+','&','!','%','?','_','|',':','-','=','\\','~','*','^','<','>','[',']','$','{','}',';','.','`','@','(',')']

def remove_non_ascii_1(text):
    return ''.join(i for i in text if ord(i) < 128)

def clean(text):
    try:
        if (text == '' or text == None):
            return ''
        text = remove_non_ascii_1(text)
        text = get_normalize_code(text,-1)
        text = re.sub('@', '', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('\n+', '\n', text)
    except Exception, e:
        print e
        print 'ERROR OF clean'
    return text.strip()


def get_normalize_code(code,max_lenghth):

    split_set = SPLIT_CHARS
    codes= _WORD_SPLIT.split(code)
    result = ''
    count_length = 0
    for c in codes:
        if (c != ''):
            if (c in split_set):
                result += ' '+c+' '
            else:
                result += c
            count_length += 1
        if(max_lenghth!=-1):
            if (count_length == max_lenghth):
                break
    result = " ".join(result.split())
    return result

