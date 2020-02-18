
'''
This Class is used for creating the FOFE-Embedding-Matrix
and for preparing the Data Sets (indexing and padding). 
It also creates necessary Objects (ID-Dicts) and other Parameters.
The objects, which are needed for Tagging, are saved in a pickle-File.
This pickle-File will later be opened with the Tagging-Constructor.
'''

import numpy as np
import itertools
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict
import pickle

from tiger_data import Tiger_Data

############## Class FOFE_Encoding ##############

class FOFE_Encoding():

	# Initialise in Training- or Tagging-Mode
	def __init__(self, *args):
		if len([*args]) == 2:
			self.init_tagging(*args)
		else:
			self.init_train(*args)

	# Init-Function for Training Mode
	def init_train(self, tiger, emb_rate, param_file):
		self.emb_rate = emb_rate
		self.char_to_id = dict()
		self.word_to_id = dict()
		self.tag_to_id = dict()
		self.max_length = max([len(s) for s in tiger.train_data[0]])

		# Process WORDS and CHARS
		i = 2	# Index 0 is used for Padding, Index 1 is for Unknown Words
		n = 1	# Index 0 is for Unknown Chars
		for word, count in tiger.word_count.items():
			if count >= 2:		# only words that occur at least twice in data
				if word not in self.word_to_id:
					self.word_to_id[word] = i
					i += 1
			for char in word:
				if char not in self.char_to_id:
					self.char_to_id[char] = n
					n += 1
		self.num_chars = len(self.char_to_id) +1
		self.vocab_size = len(self.word_to_id) +2

		# Process TAGS
		i = 1	# Index 0 is used for Padding
		for tag in tiger.tag_set:
			self.tag_to_id[tag] = i
			i += 1
		self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
		self.num_tags = len(self.tag_to_id) +1

		# Create Embedding-Matrix
		self.emb_matrix = self.get_embedding_matrix()

		# Creating padded Data-Matrices with Indices
		self.train_data = self.prepare_data(tiger.train_data[0])
		self.dev_data = self.prepare_data(tiger.dev_data[0])
		self.test_data = self.prepare_data(tiger.test_data[0])

		# Creating padded Tag-Matrices with Indices
		self.train_tags = self.get_tag_encoding(tiger.train_data[1])
		self.dev_tags = self.get_tag_encoding(tiger.dev_data[1])
		self.test_tags = self.get_tag_encoding(tiger.test_data[1])

		# Saving the Parameters
		with open(param_file, 'w') as f:
			parameters = [self.char_to_id, self.word_to_id, self.tag_to_id, self.id_to_tag, self.emb_rate, self.max_length]
			pickle.dump(parameters, open(param_file, 'wb'))

	# Init-Function for Tagging-Mode
	def init_tagging(self, data, param_file): # data is List of Lists
		self.char_to_id, self.word_to_id, self.tag_to_id, self.id_to_tag, self.emb_rate, self.max_length = pickle.load(open(param_file, 'rb'))
		self.data = self.prepare_data(data, False)

	# Translate Words to IDs and pad to longest sentence
	def prepare_data(self, data, padding = True):
		data_new = []
		for sent in data:
			word_ids = []
			for word in sent:
				if word in self.word_to_id:
					word_ids.append(self.word_to_id[word])
				else:
					word_ids.append(1)
			data_new.append(word_ids)
		if padding:
			return pad_sequences(data_new, maxlen = self.max_length, padding = 'post', value = 0)
		else:
			return data_new

	# Translate Tags to IDs and pad Tag-Sequences
	def get_tag_encoding(self, tag_data):
		data_new = []
		for tags in tag_data:
			tag_ids = []
			for tag in tags:
				tag_ids.append(self.tag_to_id[tag])
			data_new.append(tag_ids)
		return to_categorical(pad_sequences(data_new, maxlen = self.max_length, padding = 'post', value = 0), num_classes = self.num_tags)

	# Create Embedding-Matrix to integrate in Embedding-Layer
	def get_embedding_matrix(self):
		emb_matrix = np.zeros((self.vocab_size, 2*self.num_chars))
		# Create special Embedding for Unknown words (ID 0 is for Unk-Chars)
		emb_matrix[1][0] = self.emb_rate**(-5)
		emb_matrix[1][self.num_chars] = self.emb_rate**(-5)
		for word, id in self.word_to_id.items():
			vec = self.word_to_vec(word)
			if vec is not None:
				emb_matrix[id] = vec
		return emb_matrix

	# Create Numpy-Array with bidirectional FOFE-Encoding
	def word_to_vec(self, word):
		vec = np.zeros(self.num_chars)
		i = -5
		for c in word:
			if c in self.char_to_id:
				vec[self.char_to_id[c]] += self.emb_rate**i
			else:
				vec[0] += self.emb_rate**i
			i += 1
		vec_back = np.zeros(self.num_chars)
		i = -5
		for c in word[::-1]:
			if c in self.char_to_id:
				vec_back[self.char_to_id[c]] += self.emb_rate**i
			else:
				vec_back[0] += self.emb_rate**i
			i += 1
		return np.concatenate([vec, vec_back])
