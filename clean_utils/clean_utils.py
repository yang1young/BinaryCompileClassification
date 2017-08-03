#!/usr/bin/python
# coding=utf-8
import re
import sys
import mistune
reload(sys)
sys.setdefaultencoding('utf-8')

_WORD_SPLIT = re.compile(b"([,+\-&!%'._?:|=\s/\*^<>$@\[\](){}#;])")
SPLIT_CHARS = [',','+','&','!','%','?','_','|',':','-','=','\\','~','*','^','<','>','[',']','$','{','}',';','.','`','@','(',')']

def remove_blank(texts):
    new_texts = []
    for text in texts:
        if(text!='' and text!=' '):
            new_texts.append(text)
    return new_texts

def remove_non_ascii_1(text):
    return ''.join(i for i in text if ord(i) < 128)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def replace_number(text):
    codes = text.split(' ')
    codes_new = []
    for code in codes:
        if(hasNumbers(code)):
            codes_new.append('VAR')
        else:
            codes_new.append(code)
    return ' '.join(codes_new)


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

def assemble_clean(text,need_replace_number,max_length,need_reverse):
    try:
        if (text == '' or text == None):
            return ''
        text = remove_non_ascii_1(text)
        text = get_normalize_code(text,-1)
        if(need_replace_number):
            text = replace_number(text)
        text = re.sub('@', '', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('\n+', '\n', text)
        if (max_length != 0):
            if(not need_reverse):
                text = ' '.join(text.split(' ')[:max_length])+ ' ..' if len(text.split(' ')) > max_length else text
            else:
                text = ' .. ' + ' '.join(text.split(' ')[-max_length:]) if len(text.split(' ')) > max_length else text

    except Exception, e:
        print e
        print 'ERROR OF clean'
    return text.strip()

def bytecode_clean(text,max_length,need_reverse):
    text = re.sub('@', '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('\n', '', text)
    texts = text.split("$")
    new_texts = []
    for t in texts:
        line = t.strip()
        ts = [line[i:i + 2] for i in range(0, len(line), 2)]
        new_texts.append(' '.join(ts))
    text = ' $ '.join(new_texts)
    text = re.sub(' +', ' ', text)
    if (max_length != 0):
        if (not need_reverse):
            text = ' '.join(text.split(' ')[:max_length]) + ' ..' if len(text.split(' ')) > max_length else text
        else:
            text = ' .. '+' '.join(text.split(' ')[-max_length:]) if len(text.split(' ')) > max_length else text
    return text.strip()
