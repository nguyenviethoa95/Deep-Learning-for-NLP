# ------------------------------------------------
#             2.1 Creating Data Splits
# ------------------------------------------------

from keras.preprocessing.sequence import pad_sequences
import random
import os
import tensorflow
import tensorflow as tf


################################
#   modify path if necessary   #
input_file = '/DATA/data.txt'
################################

tmp_dir = '/tmp'
train_verbose = 0
pad_length = 300

def read_data(input_file):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y , data_x)), vocab

data, vocab = read_data(input_file)
vocab_size = max(vocab) + 1
random.seed(42)
random.shuffle(data)
input_len = len(data)

# train_y: a list of 20-component one-hot vectors representing newsgroups
# train_x: a list of 300-component vectors where each entry corresponds to a word ID
train_y, train_x = zip(*(data[:(input_len * 8) // 10]))
dev_y, dev_x = zip(*(data[(input_len * 8) // 10: (input_len * 9) // 10]))
test_y, test_x = zip(*(data[(input_len * 9) // 10:]))



# ------------------------------------------------
#                 2.2 A Basic CNN
# ------------------------------------------------

from keras.models import Sequential, Model
from keras.layers import *

import numpy as np
train_x, train_y = np.array(train_x), np.array(train_y)
dev_x, dev_y = np.array(dev_x), np.array(dev_y)
test_x, test_y = np.array(test_x), np.array(test_y)
print(train_x.shape, train_y.shape)

# Leave those unmodified and, if requested by the task, modify them locally in the specific task
batch_size = 64
embedding_dims = 300
epochs = 50
filters = 75
kernel_size = 3     # Keras uses a different definition where a kernel size of 3 means that 3 words are convolved at each step

model = Sequential()

model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))
####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################


# convolutional layer with 75 filter and filter size k = 2, using a RELU activation function
model.add(Conv1D(filters,kernel_size=(kernel_size), activation="relu"))
# a global max pooling layer
model.add(GlobalMaxPooling1D())
# a softmax output layer
model.add(Dense(20, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# ------------------------------------------------
#                2.3 Early Stopping
# ------------------------------------------------

####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################
import keras
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='min')

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(dev_x, dev_y),callbacks=[mc])

print('Accuracy of simple CNN: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])

loss, accuracy = model.evaluate(test_x, test_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# ------------------------------------------------
#    2.4 Experimenting with CNN Hyperparameters
# ------------------------------------------------

####################################
#                                  #
#   add your implementation here   #
#                                  #
####################################


# 2.2
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#embedding_4 (Embedding)      (None, 300, 300)          25855200
#_________________________________________________________________
#conv1d_4 (Conv1D)            (None, 298, 75)           67575
#_________________________________________________________________
#global_max_pooling1d_4 (Glob (None, 75)                0
#_________________________________________________________________
#dense_4 (Dense)              (None, 20)                1520
#=================================================================
#Total params: 25,924,295
#Trainable params: 25,924,295
#Non-trainable params: 0
#_________________________________________________________________
#Epoch 1/2
#15062/15062 [==============================] - 146s 10ms/step - loss: 2.2934 - acc: 0.4349
#Epoch 2/2
#15062/15062 [==============================] - 146s 10ms/step - loss: 0.7217 - acc: 0.8361
#Accuracy of simple CNN: 0.815189

#Testing Accuracy:  0.8072