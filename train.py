#!/usr/bin/python3

'''
When calling this File, the Training of the Network starts.
Finally, the Model and Weights will be saved (name='model2').
The Training has already been done and the respective
   Model & Weights are saved in the same folder (model1). 
The Model & Weights will be loaded within the File 'tagger.py'.
'''

import argparse
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras import optimizers

from tiger_data import Tiger_Data
from fofe_encoding import FOFE_Encoding

def save_model(model, model_name):
	# save model to json
	model_json = model.to_json()
	with open(model_name+'.json', 'w') as f:
	    f.write(model_json)
	# save model weights
	model.save_weights(model_name+'.h5')
	print('### Model has been saved')

################### Main ###################

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Training for POS-Tagger')

	parser.add_argument('--data_name', type = str, default = 'tiger2.2')

	parser.add_argument('--model_name', type = str, default = 'model2')

	parser.add_argument('--parameter_filename', type = str, default = 'fofe_parameters2')

	parser.add_argument('--fofe_factor', type = float, default = 0.5)

	parser.add_argument('--num_epochs', type = int, default = 60)

	parser.add_argument('--batch_size', type = int, default = 32)

	parser.add_argument('--hidden_size', type = int, default = 256)

	args = parser.parse_args()

	# Read and preprocess TIGER Data Set
	tiger = Tiger_Data(args.data_name)
	print("\n### TIGER Data has been read\n")

	# Create FOFE Encoding
	fofe = FOFE_Encoding(tiger, args.fofe_factor, args.parameter_filename)
	print("### FOFE Encoding has been created\n")

	# LSTM Neural Network with FOFE Embeddings
	print("### Start Training of Neural Network\n")
	model = Sequential()
	model.add(Embedding(fofe.vocab_size, 2*fofe.num_chars,
						weights = [fofe.emb_matrix],
						mask_zero = True,
						trainable = False))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(args.hidden_size, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), return_sequences=True, recurrent_dropout = 0.2)))
	model.add(TimeDistributed(Dense(fofe.num_tags, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

	earlystop = EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min', restore_best_weights = True)

	model.fit(fofe.train_data, fofe.train_tags, \
			  batch_size = args.batch_size, epochs = args.num_epochs, \
			  callbacks = [earlystop], \
			  validation_data = (fofe.dev_data, fofe.dev_tags))

	# Evaluation of Model with Test Data
	eval = model.evaluate(fofe.test_data, fofe.test_tags)
	print("\nEvaluation:")
	print("loss =", eval[0], "- Accuracy =", eval[1], "\n")

	# Saving the Model and best weights for later Tagging-Use
	save_model(model, args.model_name)
