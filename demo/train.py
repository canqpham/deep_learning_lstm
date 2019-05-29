#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:39:11 2019

@author: canqpham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.models import load_model

import os
print(os.listdir("./input"))

train = pd.read_json('./input/train.json')
test = pd.read_json('./input/test.json')

#print(train.head(20))
#new_dataframe = plt.table(test)
#plt.show()

sample_submission = pd.read_csv('./input/sample_submission.csv')
#print (pd.DataFrame(train, columns=["start_time_seconds_youtube_clip", "end_time_seconds_youtube_clip", "is_turkey"]))
print(test.columns)
print(train.columns)
print(sample_submission.head(4))

print(train.shape)
print(test.shape)

print(train[train['is_turkey']==1].index)

#from IPython.display import YouTubeVideo
#YouTubeVideo(train['vid_id'][1],start=train['start_time_seconds_youtube_clip'][1],end=train['end_time_seconds_youtube_clip'][1])

print(train['audio_embedding'].head())

#see the possible list lengths of the first dimension
print("train's audio_embedding can have this many frames: "+ str(train['audio_embedding'].apply(lambda x: len(x)).unique())) 
print("test's audio_embedding can have this many frames: "+ str(test['audio_embedding'].apply(lambda x: len(x)).unique())) 

#see the possible list lengths of the first element
print("each frame can have this many features: "+str(train['audio_embedding'].apply(lambda x: len(x[0])).unique()))

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, RNN, BatchNormalization, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#split the training data to have a validation set
train_train, train_val = train_test_split(train)
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)

#Define a basic LSTM model
model = Sequential()
model.add(BatchNormalization(input_shape=(10, 128)))
model.add(Dropout(.5))
model.add(Bidirectional(RNN(128, activation='relu')))
model.add(Dense(1, activation='sigmoid'))

#maybe there is something better to use, but let's use binary_crossentropy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#fit on a portion of the training data, and validate on the rest
model.fit(x_train, y_train,
          batch_size=300,
          nb_epoch=4,validation_data=(x_val, y_val))

model.save('./turkeydetection-RNN.h5')
# Get accuracy of model on validation data. It's not AUC but it's something at least!
score, acc = model.evaluate(x_val, y_val, batch_size=300)
print('Test accuracy:', acc)

'''model1 = load_model('./turkeydetection.h5')
test_data = [k for k in test['audio_embedding']]
submission = model1.predict_classes(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})



print(submission.head(20)'''
