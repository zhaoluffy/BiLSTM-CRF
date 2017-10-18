# BiLSTM-CRF

Named Entity Recognition with Tensorflow 
This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings).

State-of-the-art performance (F1 score between 90 and 91).

# DataSet
datasets contain four different types of named entities: locations, persons, organizations, and miscellaneous entities that
do not belong in any of the three previous categories, annotatin format like:

TAKE NNP I-NP O
OVER IN I-PP O
AT NNP I-NP O
TOP NNP I-NP O
AFTER NNP I-NP O
INNINGS NNP I-NP O
VICTORY NN I-NP O

