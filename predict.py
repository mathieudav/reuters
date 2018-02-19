# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 03:35:11 2018

@author: Utilisateur
"""
from utils import predict_labels
from nltk.corpus import reuters
import pickle

text = reuters.raw(reuters.fileids()[2])

with open('variables/word_index.pkl', 'rb') as f:
    [word_index] = pickle.load(f)
    
labels = predict_labels(text, word_index, language= 'english', MAX_SEQUENCE_LENGTH = 500, EMBEDDING_DIM = 300)

print(labels)
i = 0;
for label in labels:
    if label > 0.5:
        print(reuters.categories()[i])
    i +=1