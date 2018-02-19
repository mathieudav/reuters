# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 03:44:33 2018

@author: Mathieu Daviet
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import reuters
import numpy as np
import os
from keras.models import model_from_json
from keras.layers import Embedding, Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.models import Model

def construct_input(texts, MAX_SEQUENCE_LENGTH=500):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, word_index;


def construct_output(labels):
    labelsnp = np.zeros((len(reuters.fileids()), 90), dtype=int)
    i = 0
    for labelsdoc in labels:
        for label in labelsdoc:
            labelsnp[i][reuters.categories().index(label)] = 1
        i +=1
        
    return labelsnp;


def construct_embedding_matrix(word_index, EMBEDDING_DIM, PATH_FASTTEXT='embeddings/', language='english'):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    if language == 'french':
        nameembedding = 'wiki.multi.fr.vec'
    elif language == 'german':
        nameembedding = 'wiki.multi.de.vec'
    else:
        nameembedding = 'wiki.multi.en.vec'
        
        
    f = open(os.path.join(PATH_FASTTEXT, nameembedding), encoding="utf8")
    
    for i, line in enumerate(f):
        values = line.split(' ')
        word = values[0]
        if word in word_index:
            embedding_matrix[word_index[word]-1] = np.asarray(values[1:], dtype='float32')
    f.close()
        
    return embedding_matrix;





def predict_labels(text, word_indexp, language= 'english', MAX_SEQUENCE_LENGTH = 500, EMBEDDING_DIM = 300):
    #Tokenisation et séquençage et padding pour obtenir l'input
    data, word_index = construct_input(text, MAX_SEQUENCE_LENGTH)
    #Crétion de l'embedding matrice
    word_index = word_indexp
    embedding_matrix = construct_embedding_matrix(word_index, EMBEDDING_DIM, PATH_FASTTEXT='embeddings/', language=language)
    
    #Création du modèle
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(90, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.load_weights("variables/model.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    labels = model.predict(data)
    
    return labels[0];