import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn
import clean_utils.clean_utils as cu
logging.getLogger().setLevel(logging.INFO)

def clean_str(s,is_bytecode,need_replace_number):
	if(is_bytecode):
		str = cu.bytecode_clean(s,150)
	else:
		str = cu.assemble_clean(s,need_replace_number,150)
	return str

def load_embeddings(vocabulary):
	word_embeddings = {}
	for word in vocabulary:
		word_embeddings[word] = np.random.uniform(-0.25, 0.25, 200)
	return word_embeddings

def pad_sentences(sentences, padding_word="<PAD/>", sequence_length = 100):
	"""Pad setences during training or prediction"""

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			#logging.info('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences,max_word):
	word_counts = Counter(itertools.chain(*sentences))
	word_list = word_counts.most_common()[:max_word-1]
	vocabulary_inv = [word[0] for word in word_list]
	vocabulary_inv.append('UKN')
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

def get_index(vocabulary,word):
	if(word in vocabulary):
		index = vocabulary[word]
	else:
		index = vocabulary['UKN']
	return index

def load_data(filename,max_length,max_word,need_pad,is_bytecode,need_replace_number):

	df = pd.read_csv(filename, sep='@', header=None, encoding='utf8', engine='python')
	selected = ['tag', 'assemble', 'byte']
	df.columns = selected

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	code_index = 1
	if (is_bytecode):
		code_index = 2

	x_raw = df[selected[code_index]].apply(lambda x: clean_str(x,is_bytecode,need_replace_number).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	if(need_pad):
		x_raw = pad_sentences(x_raw,max_length)
	vocabulary, vocabulary_inv = build_vocab(x_raw,max_word)

	x = np.array([[get_index(vocabulary,word) for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)
	return x, y, vocabulary, vocabulary_inv, df, labels,label_dict


def load_dev_test_data(filename,max_length, vocabulary,label_dict,need_pad,is_bytecode,need_replace_number):
	df = pd.read_csv(filename, sep='@', header=None, encoding='utf8', engine='python')
	selected = ['tag', 'assemble', 'byte']
	df.columns = selected
	code_index = 1
	if (is_bytecode):
		code_index = 2
	x_raw = df[selected[code_index]].apply(lambda x: clean_str(x, is_bytecode, need_replace_number).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
	if(need_pad):
		x_raw = pad_sentences(x_raw, max_length)
	x = np.array([[get_index(vocabulary,word) for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)
	return x,y

if __name__ == "__main__":
	train_file = './data/train.csv.zip'
	#load_data(train_file,100,10000)
