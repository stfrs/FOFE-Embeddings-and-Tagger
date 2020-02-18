#!/usr/bin/python3

'''
This File has to be called with an already trained Model and Weights.
Those are already predefined as default.
The Programm predicts the Tags of Sentences, which have to saved in a Text File.
The Text File ('test_text.txt') given in this folder is used as default.
It can be changed or extended with Sentences to tag.
'''

import argparse
from keras.models import model_from_json
import json
import numpy as np
from nltk.tokenize import word_tokenize

from fofe_encoding import FOFE_Encoding

############### Function ###############

def load_model(model_name):
	# Load json File and create Model
	with open(model_name+'.json') as f:
		model_json = f.read()
	model = model_from_json(model_json)
	# Load Weights into new Model
	model.load_weights(model_name+'.h5')
	print("\n### Model has been loaded\n")
	return model

############### Main ###############

if __name__ == "__main__":
   
	parser = argparse.ArgumentParser(description='POS-Tagging with existing Model')

	parser.add_argument('--text_file', type = str, default = 'test_text.txt',
						help = 'Name of Textfile with Sentences, seperated by Newline')

	parser.add_argument('--model_name', type = str, default = 'model1')

	parser.add_argument('--parameter_file', type = str, default = 'fofe_parameters')

	args = parser.parse_args()

	# Read Text-File and tokenize Sentences
	with open(args.text_file) as f:
		sentences = []
		for line in f:
			sentences.append(word_tokenize(line.rstrip()))

	# Translate Sentences into FOFE-Embeddings
	fofe = FOFE_Encoding(sentences, args.parameter_file)

	# Load Tagger-Model
	model = load_model(args.model_name)

	# Predict the POS-Tags for the given Data
	for words, ids in zip(sentences, fofe.data):
		predicted_tags = [fofe.id_to_tag[id] for id in model.predict_classes(ids).squeeze()]
		for word, tag in zip(words, predicted_tags):
			print(word, tag, sep = '\t')
		print()
