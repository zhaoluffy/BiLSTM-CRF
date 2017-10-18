# BiLSTM-CRF

Named Entity Recognition with Tensorflow 
This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings).

State-of-the-art performance (F1 score between 90 and 91).

## DataSet
datasets contain four different types of named entities: locations, persons, organizations, and miscellaneous entities that
do not belong in any of the three previous categories, annotatin format like:

		TAKE NNP I-NP O
		OVER IN I-PP O
		AT NNP I-NP O
		TOP NNP I-NP O
		AFTER NNP I-NP O
		INNINGS NNP I-NP O
		VICTORY NN I-NP O

## Model
### BiLSTM-CRF(chars embeddings using BiLSTM)
* get final states of a bi-lstm on character embeddings to get a character-based representation of each word
### BiLSTM-CRF(chars embeddings using Gate CNN)
* reduce mean features of gate cnn on character embeddings to get a character-based representation of each word

* concat word embedding feature and chars feature of the word
* run a bi-lstm on each sentence to extract contextual representation of each word
* decode with a linear chain CRF

## Getting started
	Download Glove vector, and put it into data/golve.6B path
	run build_vocab.py to build vocabulary of data and get trimmed vector using pretrained glove vector
	run ner.py start train model from scrach
