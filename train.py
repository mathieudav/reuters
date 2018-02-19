# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 03:19:27 2018

@author: Mathieu Daviet
"""

from nltk.corpus import reuters
import numpy as np
from utils import construct_input, construct_output, construct_embedding_matrix
from keras.layers import Embedding, Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.models import Model
import pickle

#Paramètres

MAX_SEQUENCE_LENGTH = 500 #pour chaque texte
EMBEDDING_DIM = 300

texts = []  # liste des textes dans le dataset
labels = []  # liste des labels associés à chaque texte

for doc in reuters.fileids():
    texts.append(reuters.raw(doc))
    labels.append(reuters.categories(doc))


#Tokenisation et séquençage et padding pour obtenir l'input
data, word_index = construct_input(texts, MAX_SEQUENCE_LENGTH)

#Sauvegarder word_index pour la suite
with open('variables/word_index.pkl', 'wb') as f:
    pickle.dump([word_index], f)

#Construction des vecteurs de sortie
labelsnp = construct_output(labels)

#Split alétoire train set et validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labelsnp = labelsnp[indices]
nb_validation_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labelsnp[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labelsnp[-nb_validation_samples:]

#Crétion de l'embedding matrice
embedding_matrix = construct_embedding_matrix(word_index, EMBEDDING_DIM, PATH_FASTTEXT='embeddings/', language='english')

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
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#Entrainement du modèle
model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          validation_data=(x_val, y_val))

#Evaluation finale de la précision
loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
print('Accuracy: %f' % (accuracy*100))

model.save_weights("variables/model.h5")